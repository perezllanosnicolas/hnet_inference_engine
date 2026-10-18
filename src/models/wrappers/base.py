"""
Interfaz común de los modelos del motor de inferencia.

``BaseModelWrapper`` concentra en un único lugar el bucle de generación, el
muestreo, las condiciones de parada y la medición de tiempos, de modo que H-Net
y el baseline Transformer se evalúan con exactamente el mismo código. Cada
subclase implementa únicamente las operaciones dependientes del modelo: carga,
tokenización, reserva del estado, prefill y paso de decodificación.

Contabilidad de unidades: el límite de generación se expresa en bytes UTF-8
(``max_new_bytes``), comparables entre modelos. El resultado conserva tanto los
bytes como las unidades nativas (bytes en H-Net, tokens BPE en Qwen) para poder
traducir unas en otras en el informe.

Medición de tiempos: se emplean eventos CUDA, que se registran de forma
asíncrona en el stream y no introducen sincronizaciones adicionales; la única
sincronización debida a la instrumentación se realiza tras finalizar la
generación. El bucle sí sincroniza una vez por paso al leer la unidad generada
(``.item()``), lo cual es inherente a la evaluación de las condiciones de parada
y afecta por igual a todos los modelos.
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch


@dataclass
class GenerationConfig:
    """Parámetros de generación y muestreo."""

    max_new_bytes: int
    temperature: float = 1.0          # 0.0 equivale a decodificación voraz (greedy)
    top_k: int = 0                    # 0 desactiva el filtrado top-k
    top_p: float = 1.0                # 1.0 desactiva el filtrado nucleus
    repetition_penalty: float = 1.0   # 1.0 desactiva la penalización (formulación CTRL)
    stop_sequences: list[str] = field(default_factory=list)
    stop_unit_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False          # permite generar longitudes fijas en benchmarks
    seed: int | None = None
    measure_time: bool = True


@dataclass
class GenerationResult:
    """Salida de ``generate`` con contabilidad en bytes y en unidades nativas."""

    text: str
    output_ids: torch.Tensor          # (n_new_units,) en CPU
    unit: str
    n_prompt_units: int
    n_prompt_bytes: int
    n_new_units: int
    n_new_bytes: int
    stop_reason: Literal["max_bytes", "eos", "stop_unit", "stop_sequence"]
    prefill_time_s: float | None      # desde el inicio del prefill hasta la primera unidad
    decode_time_s: float | None       # resto de unidades tras la primera
    extra: dict = field(default_factory=dict)

    @property
    def bytes_per_unit(self) -> float:
        """Bytes medios por unidad nativa generada (1.0 en H-Net)."""
        return self.n_new_bytes / self.n_new_units if self.n_new_units else 0.0

    @property
    def prefill_units_per_s(self) -> float | None:
        if not self.prefill_time_s:
            return None
        return self.n_prompt_units / self.prefill_time_s

    @property
    def decode_units_per_s(self) -> float | None:
        if not self.decode_time_s or self.n_new_units < 2:
            return None
        return (self.n_new_units - 1) / self.decode_time_s

    @property
    def decode_bytes_per_s(self) -> float | None:
        units_per_s = self.decode_units_per_s
        if units_per_s is None:
            return None
        return units_per_s * self.bytes_per_unit


class _Timer:
    """Cronómetro basado en eventos CUDA (o ``perf_counter`` en CPU)."""

    def __init__(self, device: torch.device, enabled: bool):
        self.enabled = enabled
        self.cuda = enabled and device.type == "cuda"
        self.marks = []

    def mark(self):
        if not self.enabled:
            return
        if self.cuda:
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.marks.append(event)
        else:
            self.marks.append(time.perf_counter())

    def intervals(self) -> list[float | None]:
        """Devuelve los intervalos entre marcas consecutivas, en segundos."""
        if not self.enabled:
            return [None, None]
        if self.cuda:
            self.marks[-1].synchronize()
            return [a.elapsed_time(b) / 1000.0 for a, b in zip(self.marks, self.marks[1:])]
        return [b - a for a, b in zip(self.marks, self.marks[1:])]


class BaseModelWrapper(ABC):
    """Clase base de los envoltorios de modelo del motor."""

    unit: Literal["byte", "token"]

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.device = torch.device(device)
        self.dtype = dtype
        self.model: torch.nn.Module | None = None

    # ------------------------------------------------------------------
    # Operaciones dependientes del modelo
    # ------------------------------------------------------------------

    @abstractmethod
    def load(self, checkpoint: str, config: str | None = None) -> None:
        """Carga los pesos en ``self.model`` con el dispositivo y la precisión fijados."""

    @abstractmethod
    def encode(self, text: str) -> torch.Tensor:
        """Convierte texto en un tensor ``(1, L)`` de unidades nativas en ``self.device``."""

    @abstractmethod
    def unit_to_bytes(self, unit_id: int) -> bytes:
        """Devuelve los bytes que representa una unidad (``b""`` para unidades especiales)."""

    @property
    @abstractmethod
    def eos_unit_ids(self) -> list[int]:
        """Identificadores de fin de secuencia del modelo."""

    @abstractmethod
    def allocate_state(self, batch_size: int, max_seqlen: int) -> Any:
        """Reserva el estado de inferencia (caché KV y/o estados recurrentes)."""

    @abstractmethod
    def prefill(self, input_ids: torch.Tensor, state: Any) -> torch.Tensor:
        """Procesa el prompt ``(B, L)`` y devuelve los logits ``(B, V)`` de la última posición."""

    @abstractmethod
    def decode_step(self, last_ids: torch.Tensor, state: Any) -> torch.Tensor:
        """Avanza una unidad ``(B, 1)`` y devuelve los logits ``(B, V)`` resultantes."""

    def collect_extra(self, state: Any) -> dict:
        """Métricas propias del modelo tras la generación (p. ej. tasa de compresión)."""
        return {}

    # ------------------------------------------------------------------
    # Operaciones comunes
    # ------------------------------------------------------------------

    def decode(self, ids) -> str:
        """Reconstruye el texto concatenando los bytes de cada unidad."""
        ids = ids.tolist() if isinstance(ids, torch.Tensor) else ids
        data = b"".join(self.unit_to_bytes(int(i)) for i in ids)
        return data.decode("utf-8", errors="replace")

    @property
    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters())

    def segment(self, text: str) -> list[list[int]]:
        """
        Devuelve, por nivel de segmentación, los desplazamientos en bytes donde comienza cada unidad.

        La implementación por defecto contiene un único nivel: las unidades
        nativas producidas por ``encode`` (tokens BPE en Qwen). Las unidades
        especiales, que no representan bytes, se omiten. H-Net la redefine para
        devolver las fronteras de chunk de cada etapa.
        """
        starts, offset = [], 0
        for unit_id in self.encode(text)[0].tolist():
            data = self.unit_to_bytes(int(unit_id))
            if data:
                starts.append(offset)
                offset += len(data)
        return [starts]

    @staticmethod
    def apply_repetition_penalty(
        logits: torch.Tensor, seen: torch.Tensor, penalty: float
    ) -> torch.Tensor:
        """Penaliza las unidades ya vistas: divide los logits positivos y multiplica los negativos."""
        penalized = torch.where(logits > 0, logits / penalty, logits * penalty)
        return torch.where(seen, penalized, logits)

    @staticmethod
    def sample(
        logits: torch.Tensor, config: GenerationConfig, generator: torch.Generator | None = None
    ) -> torch.Tensor:
        """Selecciona la siguiente unidad ``(B, 1)`` a partir de logits ``(B, V)`` en float32."""
        if config.temperature == 0.0:
            return logits.argmax(dim=-1, keepdim=True)

        logits = logits / config.temperature

        if config.top_k > 0:
            k = min(config.top_k, logits.size(-1))
            threshold = torch.topk(logits, k, dim=-1).values[..., -1:]
            logits = logits.masked_fill(logits < threshold, float("-inf"))

        if config.top_p < 1.0:
            sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
            cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Se elimina toda unidad cuya masa acumulada previa ya supera top_p
            remove = (cum_probs - sorted_logits.softmax(dim=-1)) > config.top_p
            sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=generator)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        config: GenerationConfig,
        on_bytes: Callable[[bytes], None] | None = None,
    ) -> GenerationResult:
        """
        Genera texto a partir de un único prompt (B = 1).

        La generación termina al alcanzar ``max_new_bytes`` (la última unidad
        puede exceder ligeramente el límite si ocupa varios bytes), al producirse
        una unidad de fin de secuencia o de parada, o al completarse una
        secuencia de parada. Las secuencias de parada se excluyen de ``text``
        pero se contabilizan en ``n_new_bytes``.

        ``on_bytes``, si se indica, recibe los bytes de cada unidad generada en
        cuanto se conocen (las unidades de fin de secuencia y de parada no se
        notifican). Permite el streaming sin duplicar el bucle; se invoca tras
        la sincronización ya existente de cada paso, por lo que no añade
        sincronizaciones, aunque su coste de CPU se incluye en ``decode_time_s``.
        Una excepción lanzada desde ``on_bytes`` interrumpe la generación.
        """
        if self.model is None:
            raise RuntimeError("El modelo no está cargado: debe invocarse load() antes de generate().")
        if config.max_new_bytes < 1:
            raise ValueError("max_new_bytes debe ser al menos 1.")

        generator = None
        if config.seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(config.seed)

        input_ids = self.encode(prompt)
        n_prompt_units = input_ids.shape[1]
        n_prompt_bytes = len(prompt.encode("utf-8"))

        stop_ids = set(config.stop_unit_ids)
        eos_ids = set() if config.ignore_eos else set(self.eos_unit_ids)
        stop_seqs = [s.encode("utf-8") for s in config.stop_sequences if s]
        use_penalty = config.repetition_penalty != 1.0

        # Cota superior válida para ambos modelos: toda unidad no especial ocupa al menos 1 byte
        state = self.allocate_state(batch_size=1, max_seqlen=n_prompt_units + config.max_new_bytes)

        timer = _Timer(self.device, config.measure_time)
        timer.mark()

        logits = self.prefill(input_ids, state).float()
        seen = None
        if use_penalty:
            seen = torch.zeros_like(logits, dtype=torch.bool).scatter_(1, input_ids, True)

        generated_ids: list[int] = []
        generated_bytes = bytearray()
        stop_reason = "max_bytes"
        stop_seq_len = 0

        while True:
            if use_penalty:
                logits = self.apply_repetition_penalty(logits, seen, config.repetition_penalty)
            next_ids = self.sample(logits, config, generator)
            if not generated_ids:
                timer.mark()  # primera unidad disponible: fin de la fase de prefill

            next_id = int(next_ids.item())
            generated_ids.append(next_id)

            if next_id in eos_ids:
                stop_reason = "eos"
                break
            if next_id in stop_ids:
                stop_reason = "stop_unit"
                break

            unit_bytes = self.unit_to_bytes(next_id)
            generated_bytes += unit_bytes
            if on_bytes is not None and unit_bytes:
                on_bytes(unit_bytes)
            matched = next((s for s in stop_seqs if generated_bytes.endswith(s)), None)
            if matched is not None:
                stop_reason = "stop_sequence"
                stop_seq_len = len(matched)
                break
            if len(generated_bytes) >= config.max_new_bytes:
                break

            if use_penalty:
                seen.scatter_(1, next_ids, True)
            logits = self.decode_step(next_ids, state).float()

        timer.mark()
        prefill_time, decode_time = timer.intervals()

        text_bytes = generated_bytes[: len(generated_bytes) - stop_seq_len]
        return GenerationResult(
            text=bytes(text_bytes).decode("utf-8", errors="replace"),
            output_ids=torch.tensor(generated_ids, dtype=torch.long),
            unit=self.unit,
            n_prompt_units=n_prompt_units,
            n_prompt_bytes=n_prompt_bytes,
            n_new_units=len(generated_ids),
            n_new_bytes=len(generated_bytes),
            stop_reason=stop_reason,
            prefill_time_s=prefill_time,
            decode_time_s=decode_time,
            extra=self.collect_extra(state),
        )
