"""
Esquemas de petición y respuesta de la API REST del motor de inferencia.

Las respuestas reproducen las estructuras de ``src/core/engine.py``
(``GenerationReport``, ``Segmentation``, ``ModelInfo``); los límites de los
parámetros de muestreo se validan aquí, y los dependientes de la configuración
del motor (longitud del prompt y de la generación), en el propio motor.
"""

from typing import Any

from pydantic import BaseModel, Field

from src.models.wrappers.base import GenerationConfig


class SamplingParams(BaseModel):
    """Parámetros de generación comunes a todas las peticiones."""

    max_new_bytes: int = Field(256, ge=1, description="Límite de generación en bytes UTF-8.")
    temperature: float = Field(0.7, ge=0.0, le=5.0, description="0 equivale a decodificación voraz.")
    top_k: int = Field(0, ge=0, description="0 desactiva el filtrado top-k.")
    top_p: float = Field(0.9, gt=0.0, le=1.0, description="1 desactiva el filtrado nucleus.")
    repetition_penalty: float = Field(1.0, ge=1.0, le=5.0, description="1 desactiva la penalización.")
    stop_sequences: list[str] = Field(default_factory=list, max_length=8)
    ignore_eos: bool = False
    seed: int | None = Field(0, description="Semilla del muestreo; null para no fijarla.")

    def to_config(self) -> GenerationConfig:
        return GenerationConfig(
            max_new_bytes=self.max_new_bytes,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            stop_sequences=list(self.stop_sequences),
            ignore_eos=self.ignore_eos,
            seed=self.seed,
        )


class GenerateBody(SamplingParams):
    prompt: str = Field(..., min_length=1)
    model: str | None = Field(None, description="Modelo del catálogo; null usa el modelo cargado.")
    segment: bool = Field(False, description="Incluir la segmentación de prompt + texto generado.")


class CompareBody(SamplingParams):
    prompt: str = Field(..., min_length=1)
    models: list[str] = Field(..., min_length=1, max_length=4)
    segment: bool = True


class SegmentBody(BaseModel):
    text: str = Field(..., min_length=1)
    model: str | None = None


class LoadModelBody(BaseModel):
    name: str


class ModelInfoOut(BaseModel):
    name: str
    family: str
    description: str
    cached: bool
    loaded: bool


class SegmentLevelOut(BaseModel):
    name: str
    segments: list[str]
    n_units: int
    bytes_per_unit: float


class SegmentationOut(BaseModel):
    text: str
    n_bytes: int
    levels: list[SegmentLevelOut]


class GenerationReportOut(BaseModel):
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
    extra: dict[str, Any]
    segmentation: SegmentationOut | None = None


class ErrorOut(BaseModel):
    detail: str
