"""
Benchmark de latencia y throughput de generación (batch = 1).

Para cada modelo y longitud de prompt se mide, tras un calentamiento
obligatorio con la misma configuración:

- Prefill: tiempo hasta la primera unidad generada (TTFT) y unidades/s.
- Decodificación: bytes/s y unidades nativas/s, junto con ``bytes_per_unit``
  para traducir unas en otras.

La decodificación es voraz y con ``ignore_eos`` activado, de modo que todas las
repeticiones generan exactamente ``--gen-bytes`` bytes (o los más próximos por
exceso en Qwen). En H-Net las repeticiones cronometradas se ejecutan con
``track_boundaries=False``; la tasa de compresión por etapa se obtiene en una
ejecución adicional no cronometrada. Con ``--profile-shim`` se ejecuta además
una generación con el contador del shim de alineación de ``causal_conv1d``.

Ejemplo:
    .venv/bin/python scripts/eval_throughput.py \\
        --models hnet_1stage_L hnet_2stage_L Qwen2.5-0.5B Qwen2.5-1.5B \\
        --prompt-bytes 64 256 1024 --gen-bytes 256 --repeats 5 --warmup 2
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
    parser = argparse.ArgumentParser(description="Benchmark de throughput de generación.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--prompt-bytes", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--gen-bytes", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2, help="Iteraciones de calentamiento (mínimo 1).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qwen-cache", choices=["dynamic", "static"], default="dynamic")
    parser.add_argument("--qwen-attn", choices=["flash_attention_2", "sdpa", "eager"], default=None)
    parser.add_argument("--profile-shim", action="store_true",
                        help="Cuantifica las copias del shim de alineación en H-Net.")
    parser.add_argument("--output", default=None, help="Ruta del JSON (por defecto, results/).")
    args = parser.parse_args()
    if args.warmup < 1:
        parser.error("--warmup debe ser al menos 1: el calentamiento es obligatorio.")
    if args.repeats < 1:
        parser.error("--repeats debe ser al menos 1.")
    return args


def run_record(result) -> dict:
    """Extrae de un ``GenerationResult`` los campos de una repetición."""
    return {
        "n_prompt_units": result.n_prompt_units,
        "n_new_units": result.n_new_units,
        "n_new_bytes": result.n_new_bytes,
        "stop_reason": result.stop_reason,
        "prefill_time_s": result.prefill_time_s,
        "decode_time_s": result.decode_time_s,
        "prefill_units_per_s": result.prefill_units_per_s,
        "decode_units_per_s": result.decode_units_per_s,
        "decode_bytes_per_s": result.decode_bytes_per_s,
        "bytes_per_unit": result.bytes_per_unit,
    }


def benchmark_config(wrapper, spec: str, prompt_bytes: int, args) -> dict:
    prompt = metrics.make_prompt(prompt_bytes)
    config = GenerationConfig(
        max_new_bytes=args.gen_bytes, temperature=0.0, ignore_eos=True, seed=args.seed
    )

    metrics.set_seed(args.seed)
    metrics.warmup(wrapper, prompt, config, args.warmup)
    runs = [run_record(wrapper.generate(prompt, config)) for _ in range(args.repeats)]

    entry = {
        "model": spec,
        "unit": wrapper.unit,
        "prompt_bytes": len(prompt.encode("utf-8")),
        "prompt_repeated": metrics.prompt_repeats(prompt_bytes),
        "gen_bytes": args.gen_bytes,
        "runs": runs,
        "summary": {
            key: metrics.summarize([r[key] for r in runs])
            for key in ("prefill_time_s", "prefill_units_per_s", "decode_units_per_s",
                        "decode_bytes_per_s", "bytes_per_unit")
        },
        "extra": {},
    }

    if metrics.is_hnet(spec):
        # Ejecución no cronometrada para la tasa de compresión por etapa
        wrapper.track_boundaries = True
        entry["extra"] = wrapper.generate(prompt, config).extra
        wrapper.track_boundaries = False

        if args.profile_shim:
            from src.models.architectures.hnet_compat import SHIM_PROFILER

            SHIM_PROFILER.enable(timing=True)
            result = wrapper.generate(prompt, config)
            shim = SHIM_PROFILER.summary()
            SHIM_PROFILER.disable()
            shim["prefill_time_s"] = result.prefill_time_s
            if shim["copy_time_ms"] is not None and result.prefill_time_s:
                shim["fraction_of_prefill"] = shim["copy_time_ms"] / 1000.0 / result.prefill_time_s
            entry["shim"] = shim
    else:
        entry["extra"] = wrapper.collect_extra(None)

    return entry


def print_summary(results: list[dict]) -> None:
    headers = ["modelo", "prompt B", "unid. prompt", "TTFT ms", "prefill u/s",
               "decod. B/s", "+/-", "decod. u/s", "B/unid.", "compresión"]
    rows = []
    for entry in results:
        s = entry["summary"]
        ttft = s["prefill_time_s"]
        decode_bps = s["decode_bytes_per_s"]
        compression = entry["extra"].get("prefill_compression_per_stage")
        rows.append([
            entry["model"],
            entry["prompt_bytes"],
            entry["runs"][0]["n_prompt_units"],
            f"{ttft['median'] * 1000:.1f}" if ttft else None,
            f"{s['prefill_units_per_s']['median']:.0f}" if s["prefill_units_per_s"] else None,
            f"{decode_bps['median']:.1f}" if decode_bps else None,
            f"{decode_bps['std']:.1f}" if decode_bps else None,
            f"{s['decode_units_per_s']['median']:.1f}" if s["decode_units_per_s"] else None,
            f"{s['bytes_per_unit']['median']:.2f}",
            "/".join(f"{c:.2f}" for c in compression if c) if compression else None,
        ])
    print(metrics.format_table(headers, rows))
    print("Valores: mediana de las repeticiones; '+/-' es la desviación típica de bytes/s.")
    print("Compresión: bytes de prompt (con BOS) por chunk en cada etapa durante el prefill.")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        sys.exit("Se requiere una GPU CUDA para ejecutar el benchmark.")

    device = "cuda"
    dtype = metrics.default_dtype()
    payload = {
        "benchmark": "throughput",
        "environment": metrics.collect_environment(device),
        "config": {**vars(args), "dtype": str(dtype), "batch_size": 1, "decoding": "greedy",
                   "ignore_eos": True},
        "results": [],
    }
    output = args.output or metrics.default_output_path("throughput")

    for spec in args.models:
        print(f"Cargando {spec}...", flush=True)
        wrapper = metrics.build_wrapper(
            spec, device=device, dtype=dtype, track_boundaries=False,
            qwen_attn=args.qwen_attn, qwen_cache=args.qwen_cache,
        )
        print(f"  {wrapper.num_parameters / 1e6:.0f} M parámetros", flush=True)
        for prompt_bytes in args.prompt_bytes:
            print(f"  prompt de {prompt_bytes} B: calentamiento ({args.warmup}) y "
                  f"{args.repeats} repeticiones", flush=True)
            entry = benchmark_config(wrapper, spec, prompt_bytes, args)
            entry["num_parameters"] = wrapper.num_parameters
            payload["results"].append(entry)
            # Se guarda tras cada configuración para no perder resultados parciales
            metrics.save_results(output, payload)
        del wrapper
        metrics.release_memory(device)

    print()
    print_summary(payload["results"])
    if args.profile_shim:
        print()
        for entry in payload["results"]:
            if "shim" in entry:
                shim = entry["shim"]
                fraction = shim.get("fraction_of_prefill")
                print(f"Shim {entry['model']} ({entry['prompt_bytes']} B): "
                      f"{shim['aligned_copies']} copias en {shim['fwd_calls']} llamadas, "
                      f"{shim['bytes_copied'] / metrics.MIB:.2f} MiB, {shim['copy_time_ms']:.3f} ms"
                      + (f" ({fraction:.1%} del prefill)" if fraction is not None else ""))
    print(f"\nResultados guardados en {output}")


if __name__ == "__main__":
    main()
