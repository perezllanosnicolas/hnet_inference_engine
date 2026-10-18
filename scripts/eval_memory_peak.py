"""
Benchmark del pico de memoria de GPU durante la inferencia (batch = 1).

Para cada modelo se mide la memoria retenida tras la carga, desglosada en
parámetros y buffers, y, para cada combinación de
longitud de prompt y de generación, tras un calentamiento obligatorio:

- ``state``: memoria reservada por ``allocate_state`` (caché KV preasignada y
  estados recurrentes). Es nula en Qwen con caché dinámica, que crece durante
  la generación.
- ``prefill``: pico durante el procesamiento del prompt, con el estado ya
  reservado.
- ``generate``: pico de la generación completa (reserva del estado, prefill y
  decodificación), equivalente al uso real del motor.

Cada medición se realiza en un contexto ``PeakMemory`` independiente. La
métrica principal es ``max_memory_allocated`` (``*_allocated_*``); se registra
también ``max_memory_reserved`` (``*_reserved_*``). Los valores ``delta`` son
incrementos sobre la memoria ocupada por los pesos.

Ejemplo:
    .venv/bin/python scripts/eval_memory_peak.py \\
        --models hnet_1stage_L hnet_2stage_L Qwen2.5-0.5B Qwen2.5-1.5B \\
        --prompt-bytes 256 1024 4096 --gen-bytes 256
"""

import argparse
import os
import sys

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.wrappers.base import GenerationConfig  # noqa: E402
from src.utils import metrics  # noqa: E402

DEFAULT_MODELS = ["hnet_1stage_L", "hnet_2stage_L", "Qwen2.5-0.5B", "Qwen2.5-1.5B"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark del pico de memoria de GPU.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--prompt-bytes", nargs="+", type=int, default=[256, 1024, 4096])
    parser.add_argument("--gen-bytes", nargs="+", type=int, default=[256])
    parser.add_argument("--warmup", type=int, default=1, help="Iteraciones de calentamiento (mínimo 1).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qwen-cache", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--qwen-attn", choices=["flash_attention_2", "sdpa", "eager"], default=None)
    parser.add_argument("--output", default=None, help="Ruta del JSON (por defecto, results/).")
    args = parser.parse_args()
    if args.warmup < 1:
        parser.error("--warmup debe ser al menos 1: el calentamiento es obligatorio.")
    return args


def module_memory(model: torch.nn.Module) -> dict:
    """
    Desglosa la memoria del modelo cargado en parámetros y buffers, en MiB.

    Los buffers no son despreciables en el baseline: en ``transformers`` 4.44.2
    cada capa de Qwen2 precalcula sus tablas de RoPE (``cos_cached`` y
    ``sin_cached``) para ``max_position_embeddings`` posiciones.
    """
    params = {p.data_ptr(): p.numel() * p.element_size() for p in model.parameters()}
    buffers = {b.data_ptr(): b.numel() * b.element_size() for b in model.buffers()}
    return {
        "parameters_mib": round(sum(params.values()) / metrics.MIB, 2),
        "buffers_mib": round(sum(buffers.values()) / metrics.MIB, 2),
    }


@torch.inference_mode()
def measure_phases(wrapper, prompt: str, config: GenerationConfig, device: str) -> dict:
    """Mide por separado la reserva del estado, el prefill y la generación completa."""
    input_ids = wrapper.encode(prompt)
    max_seqlen = input_ids.shape[1] + config.max_new_bytes

    with metrics.PeakMemory(device) as state_mem:
        state = wrapper.allocate_state(batch_size=1, max_seqlen=max_seqlen)
    with metrics.PeakMemory(device) as prefill_mem:
        logits = wrapper.prefill(input_ids, state)
    del logits, state
    metrics.release_memory(device)

    with metrics.PeakMemory(device) as generate_mem:
        result = wrapper.generate(prompt, config)

    return {
        "n_prompt_units": input_ids.shape[1],
        "n_new_units": result.n_new_units,
        "n_new_bytes": result.n_new_bytes,
        "state": state_mem.as_dict(),
        "prefill": prefill_mem.as_dict(),
        "generate": generate_mem.as_dict(),
    }


def print_summary(models: list[dict]) -> None:
    headers = ["modelo", "carga MiB", "parám. MiB", "buffers MiB", "prompt B", "gen. B", "estado MiB",
               "pico prefill MiB", "pico gen. MiB", "+ sobre carga", "reservada MiB"]
    rows = []
    for model in models:
        for entry in model["configs"]:
            rows.append([
                model["model"],
                f"{model['weights_mib']:.0f}",
                f"{model['parameters_mib']:.0f}",
                f"{model['buffers_mib']:.0f}",
                entry["prompt_bytes"],
                entry["gen_bytes"],
                f"{entry['state']['retained_allocated_delta_mib']:.1f}",
                f"{entry['prefill']['peak_allocated_mib']:.0f}",
                f"{entry['generate']['peak_allocated_mib']:.0f}",
                f"{entry['generate']['peak_allocated_delta_mib']:.1f}",
                f"{entry['generate']['peak_reserved_mib']:.0f}",
            ])
    print(metrics.format_table(headers, rows))
    print("Métrica principal: max_memory_allocated; 'reservada' es max_memory_reserved.")
    print("'carga' es la memoria retenida tras cargar el modelo (parámetros + buffers).")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("Se requiere una GPU CUDA para ejecutar el benchmark.")

    device = "cuda"
    dtype = metrics.default_dtype()
    payload = {
        "benchmark": "memory_peak",
        "environment": metrics.collect_environment(device),
        "config": {**vars(args), "dtype": str(dtype), "batch_size": 1, "decoding": "greedy",
                   "ignore_eos": True, "memory_metric": "max_memory_allocated"},
        "results": [],
    }
    output = args.output or metrics.default_output_path("memory_peak")

    for spec in args.models:
        print(f"Cargando {spec}...", flush=True)
        metrics.release_memory(device)
        with metrics.PeakMemory(device) as load_mem:
            # Sin seguimiento de fronteras, que no forma parte del uso real del motor
            wrapper = metrics.build_wrapper(
                spec, device=device, dtype=dtype, track_boundaries=False,
                qwen_attn=args.qwen_attn, qwen_cache=args.qwen_cache,
            )
        model_entry = {
            "model": spec,
            "unit": wrapper.unit,
            "num_parameters": wrapper.num_parameters,
            "weights_mib": load_mem.as_dict()["retained_allocated_delta_mib"],
            **module_memory(wrapper.model),
            "load": load_mem.as_dict(),
            "configs": [],
        }
        payload["results"].append(model_entry)

        for prompt_bytes in args.prompt_bytes:
            prompt = metrics.make_prompt(prompt_bytes)
            for gen_bytes in args.gen_bytes:
                print(f"  prompt de {prompt_bytes} B, generación de {gen_bytes} B", flush=True)
                config = GenerationConfig(
                    max_new_bytes=gen_bytes, temperature=0.0, ignore_eos=True,
                    seed=args.seed, measure_time=False,
                )
                metrics.set_seed(args.seed)
                metrics.warmup(wrapper, prompt, config, args.warmup)
                entry = {
                    "prompt_bytes": len(prompt.encode("utf-8")),
                    "prompt_repeated": metrics.prompt_repeats(prompt_bytes),
                    "gen_bytes": gen_bytes,
                    **measure_phases(wrapper, prompt, config, device),
                }
                model_entry["configs"].append(entry)
                metrics.save_results(output, payload)

        del wrapper
        metrics.release_memory(device)

    print()
    print_summary(payload["results"])
    print(f"\nResultados guardados en {output}")


if __name__ == "__main__":
    main()
