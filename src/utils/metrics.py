"""
Utilidades comunes de los benchmarks del motor de inferencia.

Reúnen los requisitos de reproducibilidad de las mediciones: semillas fijadas,
registro del entorno (GPU, versiones y commit), calentamiento previo a toda
medición, prompts deterministas de longitud en bytes, medición de memoria en
contextos aislados y estadísticos resumidos de las repeticiones.

Métrica principal de VRAM: ``torch.cuda.max_memory_allocated``, que refleja la
memoria ocupada por tensores con independencia del allocator de caché de
PyTorch y es, por tanto, comparable entre modelos y ejecuciones. Se registra
también ``max_memory_reserved`` (memoria retenida por el proceso, incluida la
fragmentación) como referencia de la huella real en despliegue.
"""

import gc
import json
import math
import os
import platform
import random
import subprocess
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[2]
PROMPT_PATH = ROOT / "configs" / "benchmarks" / "prompt.txt"
MIB = 1024 ** 2

TRACKED_PACKAGES = (
    "torch",
    "triton",
    "flash-attn",
    "mamba-ssm",
    "causal-conv1d",
    "transformers",
    "tokenizers",
    "einops",
)


# ----------------------------------------------------------------------
# Reproducibilidad y entorno
# ----------------------------------------------------------------------


def set_seed(seed: int) -> None:
    """Fija las semillas de ``random``, PyTorch (CPU) y todas las GPU."""
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def default_dtype() -> torch.dtype:
    """bfloat16 si la GPU lo admite; en caso contrario float16 (FlashAttention rechaza fp32)."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _nvidia_driver() -> str | None:
    try:
        return subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, check=True,
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None


def collect_environment(device: torch.device | str = "cuda") -> dict:
    """Describe el hardware, el software y la revisión del código de la ejecución."""
    device = torch.device(device)
    versions = {}
    for package in TRACKED_PACKAGES:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None

    gpu = None
    if device.type == "cuda" and torch.cuda.is_available():
        props = torch.cuda.get_device_properties(device)
        gpu = {
            "name": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_mib": round(props.total_memory / MIB, 1),
            "multiprocessors": props.multi_processor_count,
            "driver": _nvidia_driver(),
        }

    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "--short", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": gpu,
        "packages": versions,
    }


# ----------------------------------------------------------------------
# Modelos y prompts
# ----------------------------------------------------------------------


def is_hnet(spec: str) -> bool:
    return Path(spec).name.lower().startswith("hnet")


def build_wrapper(
    spec: str,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
    track_boundaries: bool = False,
    qwen_attn: str | None = None,
    qwen_cache: str = "dynamic",
):
    """
    Construye y carga el envoltorio correspondiente a ``spec``.

    Los nombres que comienzan por ``hnet`` (``hnet_2stage_L``, rutas a ``.pt`` o
    repositorios ``cartesia-ai/...``) se cargan con ``HNetWrapper``; el resto, con
    ``QwenWrapper``. Si no se indica ``qwen_attn``, se emplea
    ``flash_attention_2`` con caché dinámica y ``sdpa`` con caché estática.
    """
    dtype = dtype or default_dtype()
    if is_hnet(spec):
        from src.models.wrappers.hnet_wrapper import HNetWrapper

        wrapper = HNetWrapper(device=device, dtype=dtype, track_boundaries=track_boundaries)
    else:
        from src.models.wrappers.qwen_wrapper import QwenWrapper

        attn = qwen_attn or ("sdpa" if qwen_cache == "static" else "flash_attention_2")
        wrapper = QwenWrapper(
            device=device, dtype=dtype, attn_implementation=attn, cache_implementation=qwen_cache
        )
    wrapper.load(spec)
    return wrapper


def make_prompt(n_bytes: int, path: str | os.PathLike = PROMPT_PATH) -> str:
    """
    Devuelve los primeros ``n_bytes`` bytes UTF-8 del texto de referencia.

    Si se solicitan más bytes de los disponibles, el texto se repite separado
    por un salto de línea. El corte se retrocede hasta el inicio del último
    carácter completo, de modo que el resultado puede tener hasta 3 bytes
    menos que lo solicitado si el texto contiene caracteres multibyte.
    """
    if n_bytes < 1:
        raise ValueError("La longitud del prompt debe ser al menos 1 byte.")
    base = Path(path).read_text(encoding="utf-8").strip().encode("utf-8") + b"\n"
    data = base * math.ceil(n_bytes / len(base))
    cut = n_bytes
    # Los bytes de continuación UTF-8 tienen la forma 10xxxxxx
    while 0 < cut < len(data) and (data[cut] & 0xC0) == 0x80:
        cut -= 1
    return data[:cut].decode("utf-8")


def prompt_repeats(n_bytes: int, path: str | os.PathLike = PROMPT_PATH) -> bool:
    """Indica si un prompt de ``n_bytes`` bytes requiere repetir el texto de referencia."""
    return n_bytes > len(Path(path).read_text(encoding="utf-8").strip().encode("utf-8")) + 1


# ----------------------------------------------------------------------
# Calentamiento
# ----------------------------------------------------------------------


def warmup(wrapper, prompt: str, config, n: int) -> None:
    """
    Ejecuta ``n`` generaciones completas descartando su resultado.

    Absorbe la compilación JIT y el autotuning de los kernels Triton (que
    dependen de las formas de los tensores), la inicialización perezosa de
    cuBLAS y FlashAttention y el crecimiento inicial del allocator de caché.
    Debe invocarse con la misma configuración que se va a medir.
    """
    if n < 1:
        raise ValueError("El calentamiento es obligatorio: se requiere al menos una iteración.")
    for _ in range(n):
        wrapper.generate(prompt, config)
    if wrapper.device.type == "cuda":
        torch.cuda.synchronize(wrapper.device)


# ----------------------------------------------------------------------
# Memoria
# ----------------------------------------------------------------------


def release_memory(device: torch.device | str = "cuda") -> None:
    """Libera objetos inaccesibles y devuelve al driver los bloques de caché libres."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()


class PeakMemory:
    """
    Contexto de medición de memoria de GPU.

    Al entrar se liberan los bloques libres de la caché y se reinician las
    estadísticas de pico, de modo que cada medición es independiente de las
    anteriores; al salir se sincroniza y se registran los picos. Los valores
    ``*_delta`` son el incremento sobre la memoria ya ocupada al entrar (p. ej.
    los pesos del modelo).
    """

    def __init__(self, device: torch.device | str = "cuda"):
        self.device = torch.device(device)

    def __enter__(self):
        release_memory(self.device)
        torch.cuda.reset_peak_memory_stats(self.device)
        self.base_allocated = torch.cuda.memory_allocated(self.device)
        self.base_reserved = torch.cuda.memory_reserved(self.device)
        return self

    def __exit__(self, *exc):
        torch.cuda.synchronize(self.device)
        self.peak_allocated = torch.cuda.max_memory_allocated(self.device)
        self.peak_reserved = torch.cuda.max_memory_reserved(self.device)
        self.end_allocated = torch.cuda.memory_allocated(self.device)
        return False

    def as_dict(self) -> dict:
        """Valores en MiB."""
        values = {
            "base_allocated": self.base_allocated,
            "peak_allocated": self.peak_allocated,
            "peak_allocated_delta": self.peak_allocated - self.base_allocated,
            "retained_allocated_delta": self.end_allocated - self.base_allocated,
            "base_reserved": self.base_reserved,
            "peak_reserved": self.peak_reserved,
            "peak_reserved_delta": self.peak_reserved - self.base_reserved,
        }
        return {f"{k}_mib": round(v / MIB, 2) for k, v in values.items()}


# ----------------------------------------------------------------------
# Estadísticos y resultados
# ----------------------------------------------------------------------


def _percentile(sorted_values: list[float], q: float) -> float:
    """Percentil con interpolación lineal (equivalente a ``numpy.percentile``)."""
    position = (len(sorted_values) - 1) * q
    low, high = math.floor(position), math.ceil(position)
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * (position - low)


def summarize(values: list[float | None]) -> dict | None:
    """Media, desviación típica muestral, mediana, p5, p95, mínimo y máximo."""
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    n = len(values)
    mean = sum(values) / n
    std = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1)) if n > 1 else 0.0
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "median": _percentile(values, 0.5),
        "p5": _percentile(values, 0.05),
        "p95": _percentile(values, 0.95),
        "min": values[0],
        "max": values[-1],
    }


def default_output_path(prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return ROOT / "results" / f"{prefix}_{stamp}.json"


def save_results(path: str | os.PathLike, payload: dict[str, Any]) -> Path:
    """Escribe los resultados en JSON, creando el directorio si no existe."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    return path


def format_table(headers: list[str], rows: list[list[Any]]) -> str:
    """Tabla de texto alineada para la salida por consola."""
    cells = [[str(h) for h in headers]] + [["-" if c is None else str(c) for c in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    lines = ["  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in cells]
    lines.insert(1, "  ".join("-" * w for w in widths))
    return "\n".join(lines)
