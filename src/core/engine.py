"""
Motor de inferencia: capa de servicio sobre los envoltorios de modelo.

``InferenceEngine`` gestiona el ciclo de vida de los modelos y la ejecución de
las peticiones de generación con las siguientes políticas:

- Un único modelo residente en la GPU. Al solicitar otro modelo del catálogo
  se libera el anterior y se carga el nuevo, seguido de un calentamiento que
  absorbe la compilación JIT y el autotuning de los kernels.
- Ejecución serializada: todas las operaciones sobre la GPU se despachan a un
  único hilo de trabajo, que actúa como cola FIFO. Cada petición se procesa
  con batch = 1 (``BaseModelWrapper.generate``), lo que constituye la
  referencia frente a la que se evaluarán las estrategias de batching.
- Streaming: los bytes generados se notifican al bucle de eventos a medida que
  se producen y se convierten en texto con un decodificador UTF-8 incremental
  que no parte caracteres multibyte. Se retienen los últimos bytes que podrían
  formar parte de una secuencia de parada, para no emitirlos antes de saber si
  deben excluirse.
- Segmentación: además del texto, puede devolverse la división del texto en
  unidades del modelo (chunks por etapa en H-Net, tokens BPE en Qwen).

Las operaciones asíncronas están pensadas para invocarse desde el bucle de
eventos del servidor (``src/api/app.py``).
"""

import asyncio
import codecs
import json
import threading
import time
from collections.abc import AsyncIterator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from src.models.wrappers.base import BaseModelWrapper, GenerationConfig, GenerationResult
from src.utils import metrics

QWEN_MODELS = ("Qwen2.5-0.5B", "Qwen2.5-1.5B")
WARMUP_PROMPT = "The quick brown fox jumps over the lazy dog. "
WARMUP_NEW_BYTES = 16


class EngineError(Exception):
    """Petición no válida para el estado actual del motor."""


class ModelNotAvailable(EngineError):
    """Modelo fuera del catálogo o ausente de la caché local."""


class GenerationCancelled(Exception):
    """Generación interrumpida por el cliente (desconexión del streaming)."""


# ----------------------------------------------------------------------
# Configuración y catálogo
# ----------------------------------------------------------------------


@dataclass
class EngineConfig:
    """Parámetros del motor."""

    device: str = "cuda"
    dtype: torch.dtype | None = None       # None: bfloat16 si la GPU lo admite, si no float16
    default_model: str | None = "hnet_1stage_L"
    max_prompt_bytes: int = 8192
    max_new_bytes: int = 2048
    warmup_iterations: int = 1             # 0 desactiva el calentamiento tras cargar un modelo
    allow_download: bool = False           # permite cargar modelos ausentes de la caché local


@dataclass(frozen=True)
class ModelSpec:
    """Entrada del catálogo de modelos."""

    name: str
    family: str                            # "hnet", "qwen" u otra (pruebas)
    description: str = ""


def default_catalog() -> list[ModelSpec]:
    """Checkpoints de H-Net con configuración en ``configs/hnet`` y modelos Qwen de referencia."""
    from src.models.wrappers.hnet_wrapper import CONFIG_DIR

    catalog = [
        ModelSpec(path.stem, "hnet", "H-Net (cartesia-ai), bytes con dynamic chunking")
        for path in sorted(CONFIG_DIR.glob("*.json"))
    ]
    catalog += [ModelSpec(name, "qwen", "Transformer denso, tokens BPE (baseline)") for name in QWEN_MODELS]
    return catalog


def is_cached(spec: ModelSpec) -> bool:
    """Indica si los pesos del modelo están en la caché local de Hugging Face."""
    if spec.family not in ("hnet", "qwen"):
        return True
    from huggingface_hub import try_to_load_from_cache

    if spec.family == "hnet":
        repo_id, filename = f"cartesia-ai/{spec.name}", f"{spec.name}.pt"
    else:
        repo_id, filename = f"Qwen/{spec.name}", "model.safetensors"
    return isinstance(try_to_load_from_cache(repo_id, filename), str)


def default_wrapper_factory(engine_config: EngineConfig) -> Callable[[ModelSpec], BaseModelWrapper]:
    """Construye los envoltorios reales; en H-Net se contabilizan las fronteras de chunk."""

    def factory(spec: ModelSpec) -> BaseModelWrapper:
        return metrics.build_wrapper(
            spec.name,
            device=engine_config.device,
            dtype=engine_config.dtype,
            track_boundaries=True,
        )

    return factory


# ----------------------------------------------------------------------
# Peticiones y resultados
# ----------------------------------------------------------------------


@dataclass
class GenerateRequest:
    """Petición de generación dirigida al motor."""

    prompt: str
    config: GenerationConfig
    model: str | None = None               # None: modelo actualmente cargado
    segment: bool = False                  # devolver la segmentación de prompt + texto generado


@dataclass
class SegmentLevel:
    """Segmentación de un texto en un nivel (una etapa de H-Net o los tokens de Qwen)."""

    name: str
    segments: list[str]
    n_units: int
    bytes_per_unit: float


@dataclass
class Segmentation:
    text: str
    n_bytes: int
    levels: list[SegmentLevel]


@dataclass
class GenerationReport:
    """Resultado de una generación con sus métricas, serializable a JSON."""

    model: str
    family: str
    unit: str
    text: str
    stop_reason: str
    n_prompt_bytes: int
    n_prompt_units: int
    n_new_bytes: int
    n_new_units: int
    bytes_per_unit: float
    ttft_ms: float | None
    prefill_units_per_s: float | None
    decode_bytes_per_s: float | None
    decode_units_per_s: float | None
    wall_time_s: float
    peak_vram_mib: float | None
    extra: dict = field(default_factory=dict)
    segmentation: Segmentation | None = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelInfo:
    name: str
    family: str
    description: str
    cached: bool
    loaded: bool


def build_segmentation(text: str, starts_per_level: list[list[int]], level_names: list[str]) -> Segmentation:
    """
    Divide ``text`` en segmentos a partir de los desplazamientos en bytes de cada nivel.

    Las fronteras que caen dentro de un carácter multibyte (posibles tanto en
    H-Net como en los tokens BPE) se desplazan al inicio de dicho carácter, de
    modo que cada segmento es texto UTF-8 válido. ``n_units`` conserva el
    número de unidades original del modelo.
    """
    data = text.encode("utf-8")
    levels = []
    for name, starts in zip(level_names, starts_per_level):
        snapped = set()
        for start in starts:
            if not 0 <= start < len(data):
                continue
            while start > 0 and data[start] & 0xC0 == 0x80:
                start -= 1
            snapped.add(start)
        if data:
            snapped.add(0)
        bounds = sorted(snapped) + [len(data)]
        segments = [data[a:b].decode("utf-8", errors="replace") for a, b in zip(bounds, bounds[1:])]
        n_units = len(starts)
        levels.append(
            SegmentLevel(
                name=name,
                segments=segments,
                n_units=n_units,
                bytes_per_unit=round(len(data) / n_units, 3) if n_units else 0.0,
            )
        )
    return Segmentation(text=text, n_bytes=len(data), levels=levels)


class _StreamDecoder:
    """Convierte bytes en texto de forma incremental, reteniendo ``holdback`` bytes finales."""

    def __init__(self, holdback: int):
        self.holdback = holdback
        self.buffer = bytearray()
        self.emitted = 0
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.text = ""

    def feed(self, data: bytes) -> str:
        self.buffer += data
        limit = len(self.buffer) - self.holdback
        if limit <= self.emitted:
            return ""
        chunk = self.decoder.decode(bytes(self.buffer[self.emitted:limit]))
        self.emitted = limit
        self.text += chunk
        return chunk

    def finish(self, final_text: str) -> str:
        """Devuelve el texto pendiente para completar ``final_text`` (texto sin secuencia de parada)."""
        if final_text.startswith(self.text):
            return final_text[len(self.text):]
        return ""


# ----------------------------------------------------------------------
# Motor
# ----------------------------------------------------------------------


class InferenceEngine:
    """
    Servicio de inferencia con un modelo residente y una cola FIFO de ejecución.

    ``catalog`` y ``wrapper_factory`` permiten sustituir los modelos reales por
    modelos ficticios en las pruebas.
    """

    def __init__(
        self,
        config: EngineConfig | None = None,
        catalog: list[ModelSpec] | None = None,
        wrapper_factory: Callable[[ModelSpec], BaseModelWrapper] | None = None,
    ):
        self.config = config or EngineConfig()
        self.catalog = {spec.name: spec for spec in (catalog if catalog is not None else default_catalog())}
        self._factory = wrapper_factory or default_wrapper_factory(self.config)
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="motor-gpu")
        self._wrapper: BaseModelWrapper | None = None
        self._spec: ModelSpec | None = None
        self._load_info: dict[str, Any] = {}
        self._pending = 0
        self._served = 0
        self._started_at = time.time()

    @property
    def _cuda(self) -> bool:
        return torch.device(self.config.device).type == "cuda"

    # ------------------------------------------------------------------
    # Ciclo de vida
    # ------------------------------------------------------------------

    async def start(self) -> None:
        """Carga el modelo por defecto, si se ha configurado."""
        if self._cuda and not torch.cuda.is_available():
            raise RuntimeError("No se detecta ninguna GPU CUDA: H-Net requiere FlashAttention y kernels de Mamba.")
        if self.config.default_model:
            await self.load_model(self.config.default_model)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    async def _submit(self, fn: Callable, *args) -> Any:
        """Encola ``fn`` en el hilo de la GPU y espera su resultado."""
        self._pending += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(self._executor, fn, *args)
        finally:
            self._pending -= 1

    # ------------------------------------------------------------------
    # Modelos
    # ------------------------------------------------------------------

    def available_models(self) -> list[ModelInfo]:
        current = self._spec.name if self._spec else None
        return [
            ModelInfo(spec.name, spec.family, spec.description, is_cached(spec), spec.name == current)
            for spec in self.catalog.values()
        ]

    async def load_model(self, name: str) -> dict:
        return await self._submit(self._load_sync, name)

    def resolve_model(self, name: str) -> ModelSpec:
        spec = self.catalog.get(name)
        if spec is None:
            raise ModelNotAvailable(f"El modelo '{name}' no figura en el catálogo del motor.")
        if not self.config.allow_download and not is_cached(spec):
            raise ModelNotAvailable(
                f"El modelo '{name}' no está en la caché local y la descarga está desactivada "
                "(opción --allow-download)."
            )
        return spec

    def _unload_sync(self) -> None:
        self._wrapper = None
        self._spec = None
        self._load_info = {}
        if self._cuda:
            metrics.release_memory(self.config.device)

    def _load_sync(self, name: str, notify: Callable[[dict], None] | None = None) -> dict:
        """Sustituye el modelo residente por ``name`` (sin efecto si ya está cargado)."""
        if self._spec is not None and self._spec.name == name:
            return self._load_info
        spec = self.resolve_model(name)
        if notify:
            notify({"event": "status", "message": f"Cargando el modelo {name}..."})
        self._unload_sync()

        start = time.perf_counter()
        wrapper = self._factory(spec)
        load_time = time.perf_counter() - start

        warmup_time = 0.0
        if self.config.warmup_iterations > 0:
            if notify:
                notify({"event": "status", "message": f"Calentando el modelo {name}..."})
            start = time.perf_counter()
            warm_config = GenerationConfig(
                max_new_bytes=WARMUP_NEW_BYTES, temperature=0.0, ignore_eos=True, measure_time=False
            )
            metrics.warmup(wrapper, WARMUP_PROMPT, warm_config, self.config.warmup_iterations)
            warmup_time = time.perf_counter() - start

        self._wrapper, self._spec = wrapper, spec
        self._load_info = {
            "model": name,
            "family": spec.family,
            "unit": wrapper.unit,
            "num_parameters": wrapper.num_parameters,
            "load_time_s": round(load_time, 3),
            "warmup_time_s": round(warmup_time, 3),
            "weights_mib": self._allocated_mib(),
        }
        return self._load_info

    # ------------------------------------------------------------------
    # Estado
    # ------------------------------------------------------------------

    def _allocated_mib(self) -> float | None:
        if not self._cuda:
            return None
        return round(torch.cuda.memory_allocated(self.config.device) / metrics.MIB, 1)

    def status(self) -> dict:
        """Estado del motor; no pasa por la cola, por lo que responde durante una generación."""
        gpu = None
        if self._cuda:
            device = torch.device(self.config.device)
            free, total = torch.cuda.mem_get_info(device)
            gpu = {
                "name": torch.cuda.get_device_name(device),
                "allocated_mib": self._allocated_mib(),
                "reserved_mib": round(torch.cuda.memory_reserved(device) / metrics.MIB, 1),
                "used_mib": round((total - free) / metrics.MIB, 1),
                "total_mib": round(total / metrics.MIB, 1),
            }
        return {
            "model": self._load_info or None,
            "pending_requests": self._pending,
            "served_requests": self._served,
            "uptime_s": round(time.time() - self._started_at, 1),
            "gpu": gpu,
            "limits": {
                "max_prompt_bytes": self.config.max_prompt_bytes,
                "max_new_bytes": self.config.max_new_bytes,
            },
        }

    # ------------------------------------------------------------------
    # Generación
    # ------------------------------------------------------------------

    def validate_request(self, request: GenerateRequest) -> None:
        n_bytes = len(request.prompt.encode("utf-8"))
        if n_bytes == 0:
            raise EngineError("El prompt no puede estar vacío.")
        if n_bytes > self.config.max_prompt_bytes:
            raise EngineError(
                f"El prompt ocupa {n_bytes} bytes y el máximo admitido es {self.config.max_prompt_bytes}."
            )
        if not 1 <= request.config.max_new_bytes <= self.config.max_new_bytes:
            raise EngineError(f"max_new_bytes debe estar entre 1 y {self.config.max_new_bytes}.")

    def _active(self, model: str | None, notify: Callable[[dict], None] | None) -> BaseModelWrapper:
        if model is not None:
            self._load_sync(model, notify)
        if self._wrapper is None:
            raise EngineError("No hay ningún modelo cargado.")
        return self._wrapper

    def _segment_sync(self, wrapper: BaseModelWrapper, text: str) -> Segmentation:
        starts = wrapper.segment(text)
        if wrapper.unit == "byte":
            names = [f"Etapa {i + 1}" for i in range(len(starts))]
        else:
            names = ["Tokens BPE"]
        return build_segmentation(text, starts, names)

    def _generate_sync(
        self,
        request: GenerateRequest,
        on_bytes: Callable[[bytes], None] | None = None,
        notify: Callable[[dict], None] | None = None,
    ) -> GenerationReport:
        wrapper = self._active(request.model, notify)
        if notify:
            notify({"event": "start", "model": self._spec.name, "unit": wrapper.unit})
        if self._cuda:
            torch.cuda.reset_peak_memory_stats(self.config.device)

        start = time.perf_counter()
        result: GenerationResult = wrapper.generate(request.prompt, request.config, on_bytes=on_bytes)
        wall_time = time.perf_counter() - start
        peak = round(torch.cuda.max_memory_allocated(self.config.device) / metrics.MIB, 1) if self._cuda else None

        segmentation = self._segment_sync(wrapper, request.prompt + result.text) if request.segment else None
        self._served += 1
        return GenerationReport(
            model=self._spec.name,
            family=self._spec.family,
            unit=result.unit,
            text=result.text,
            stop_reason=result.stop_reason,
            n_prompt_bytes=result.n_prompt_bytes,
            n_prompt_units=result.n_prompt_units,
            n_new_bytes=result.n_new_bytes,
            n_new_units=result.n_new_units,
            bytes_per_unit=round(result.bytes_per_unit, 3),
            ttft_ms=_round(result.prefill_time_s, 1000.0),
            prefill_units_per_s=_round(result.prefill_units_per_s),
            decode_bytes_per_s=_round(result.decode_bytes_per_s),
            decode_units_per_s=_round(result.decode_units_per_s),
            wall_time_s=round(wall_time, 3),
            peak_vram_mib=peak,
            extra=result.extra,
            segmentation=segmentation,
        )

    async def generate(self, request: GenerateRequest) -> GenerationReport:
        self.validate_request(request)
        return await self._submit(self._generate_sync, request)

    async def segment(self, text: str, model: str | None = None) -> Segmentation:
        if not text:
            raise EngineError("El texto no puede estar vacío.")
        limit = self.config.max_prompt_bytes + self.config.max_new_bytes
        if len(text.encode("utf-8")) > limit:
            raise EngineError(f"El texto supera el máximo de {limit} bytes.")
        return await self._submit(lambda: self._segment_sync(self._active(model, None), text))

    async def stream(self, request: GenerateRequest) -> AsyncIterator[dict]:
        """
        Genera en streaming. Produce eventos ``status``, ``start``, ``delta``
        (texto parcial) y, al terminar, ``done`` con el ``GenerationReport``
        completo o ``error`` con el motivo. Si el consumidor abandona la
        iteración, la generación en curso se interrumpe en el siguiente paso.
        """
        self.validate_request(request)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancelled = threading.Event()
        # on_bytes se invoca antes de comprobar las secuencias de parada, por lo que se retiene
        # la longitud completa de la más larga: la coincidencia puede terminar en el byte recibido
        holdback = max((len(s.encode("utf-8")) for s in request.config.stop_sequences if s), default=0)
        decoder = _StreamDecoder(holdback)

        def notify(event: dict) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)

        def on_bytes(data: bytes) -> None:
            if cancelled.is_set():
                raise GenerationCancelled()
            text = decoder.feed(data)
            if text:
                notify({"event": "delta", "text": text})

        finished = object()
        task = asyncio.ensure_future(self._submit(self._generate_sync, request, on_bytes, notify))
        # La finalización de la tarea se encola tras los eventos ya emitidos por el hilo de la GPU
        task.add_done_callback(lambda _: queue.put_nowait(finished))
        try:
            while (event := await queue.get()) is not finished:
                yield event
            try:
                report = task.result()
            except EngineError as exc:
                yield {"event": "error", "message": str(exc)}
                return
            except Exception as exc:  # error interno del modelo (p. ej. memoria insuficiente)
                yield {"event": "error", "message": f"Error interno durante la generación: {exc}"}
                return
            tail = decoder.finish(report.text)
            if tail:
                yield {"event": "delta", "text": tail}
            yield {"event": "done", "report": report.as_dict()}
        finally:
            if not task.done():
                cancelled.set()
                task.cancel()

    async def compare(
        self, prompt: str, models: list[str], config: GenerationConfig, segment: bool = True
    ) -> AsyncIterator[dict]:
        """
        Ejecuta el mismo prompt con la misma configuración sobre varios modelos, uno tras otro.

        Cada evento de ``stream`` se etiqueta con el índice y el nombre del
        modelo; al terminar se emite ``end``. Los modelos se cargan en orden,
        por lo que el último queda residente.
        """
        if not models:
            raise EngineError("Debe indicarse al menos un modelo.")
        for name in models:
            self.resolve_model(name)
        for index, name in enumerate(models):
            yield {"event": "model", "index": index, "model": name}
            request = GenerateRequest(prompt=prompt, config=config, model=name, segment=segment)
            async for event in self.stream(request):
                yield {**event, "index": index, "model": name}
        yield {"event": "end"}


def _round(value: float | None, scale: float = 1.0, digits: int = 2) -> float | None:
    return None if value is None else round(value * scale, digits)


def to_sse(event: dict) -> str:
    """Serializa un evento en formato Server-Sent Events."""
    payload = {k: v for k, v in event.items() if k != "event"}
    return f"event: {event['event']}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"
