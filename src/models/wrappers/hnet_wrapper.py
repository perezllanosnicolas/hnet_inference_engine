"""
Envoltorio de H-Net para el motor de inferencia.

La importación de ``hnet_compat`` debe preceder a cualquier importación de
``hnet``: aplica los shims de compatibilidad y registra la ruta del paquete.

La carga reproduce la del repositorio oficial (``goombalab/hnet``,
``generate.py``): configuración JSON + ``state_dict`` en ``.pt``, y prompt
precedido del byte BOS (254). El byte 255 actúa como EOS.
"""

import json
import os
from pathlib import Path
from typing import Any

import torch

from src.models.architectures import hnet_compat  # noqa: F401
from src.models.wrappers.base import BaseModelWrapper

from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
from hnet.models.mixer_seq import HNetForCausalLM
from hnet.utils.tokenizers import ByteTokenizer

CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "hnet"
HF_ORGANIZATION = "cartesia-ai"


def resolve_checkpoint(checkpoint: str) -> tuple[str, str]:
    """
    Resuelve un checkpoint a una ruta local y a su nombre de modelo.

    Se admite una ruta a un ``.pt``, un identificador de Hugging Face
    (``cartesia-ai/hnet_2stage_L``) o el nombre del modelo (``hnet_2stage_L``);
    en los dos últimos casos el archivo se descarga o se toma de la caché local.
    """
    if os.path.isfile(checkpoint):
        return checkpoint, Path(checkpoint).stem

    from huggingface_hub import hf_hub_download

    repo_id = checkpoint if "/" in checkpoint else f"{HF_ORGANIZATION}/{checkpoint}"
    name = repo_id.split("/")[-1]
    return hf_hub_download(repo_id, f"{name}.pt"), name


def load_hnet_config(config_path: str | os.PathLike) -> HNetConfig:
    """Construye un ``HNetConfig`` a partir de un JSON del repositorio oficial."""
    with open(config_path, "r") as f:
        config = json.load(f)
    attn_cfg = AttnConfig(**config.pop("attn_cfg"))
    ssm_cfg = SSMConfig(**config.pop("ssm_cfg"))
    return HNetConfig(**config, attn_cfg=attn_cfg, ssm_cfg=ssm_cfg)


class HNetWrapper(BaseModelWrapper):
    """
    Envoltorio de ``HNetForCausalLM``.

    Con ``track_boundaries=True`` se contabilizan, sin sincronizar con la CPU,
    las fronteras de chunk de cada etapa durante el prefill y la decodificación;
    los resultados se exponen en ``GenerationResult.extra``. Puede desactivarse
    en los benchmarks de throughput para eliminar su coste (un kernel por etapa
    y paso).
    """

    unit = "byte"

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        add_bos: bool = True,
        track_boundaries: bool = True,
    ):
        super().__init__(device=device, dtype=dtype)
        self.tokenizer = ByteTokenizer()
        self.add_bos = add_bos
        self.track_boundaries = track_boundaries
        self.model_name: str | None = None
        self.config: HNetConfig | None = None
        self._prefill_chunks: list[torch.Tensor] = []
        self._prefill_length = 0
        self._decode_chunks: torch.Tensor | None = None
        self._decode_steps = 0

    @property
    def num_stages(self) -> int:
        """Número de etapas con dynamic chunking (1 en ``1stage``, 2 en ``2stage``)."""
        return len(self.config.d_model) - 1

    def load(self, checkpoint: str, config: str | None = None) -> None:
        checkpoint_path, name = resolve_checkpoint(checkpoint)
        config_path = config or CONFIG_DIR / f"{name}.json"
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"No se encuentra la configuración de '{name}' en {config_path}; "
                "debe indicarse explícitamente mediante el argumento config."
            )

        self.config = load_hnet_config(config_path)
        self.model_name = name

        model = HNetForCausalLM(self.config, device=self.device, dtype=self.dtype)
        # mmap evita materializar en RAM el checkpoint completo (fp32) antes de copiarlo a la GPU
        state_dict = torch.load(checkpoint_path, map_location="cpu", mmap=True, weights_only=True)
        model.load_state_dict(state_dict)
        del state_dict
        self.model = model.eval()

    def encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode([text], add_bos=self.add_bos)[0]["input_ids"]
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def unit_to_bytes(self, unit_id: int) -> bytes:
        # BOS (254) y EOS (255) no son bytes UTF-8 válidos y se tratan como unidades especiales
        if unit_id in (self.tokenizer.bos_idx, self.tokenizer.eos_idx):
            return b""
        return bytes([unit_id])

    @property
    def eos_unit_ids(self) -> list[int]:
        return [self.tokenizer.eos_idx]

    def allocate_state(self, batch_size: int, max_seqlen: int) -> Any:
        self._prefill_chunks = []
        self._prefill_length = 0
        self._decode_steps = 0
        self._decode_chunks = (
            torch.zeros(self.num_stages, dtype=torch.long, device=self.device)
            if self.track_boundaries
            else None
        )
        return self.model.allocate_inference_cache(batch_size, max_seqlen, dtype=self.dtype)

    def prefill(self, input_ids: torch.Tensor, state: Any) -> torch.Tensor:
        mask = torch.ones_like(input_ids, dtype=torch.bool)
        output = self.model(input_ids, mask=mask, inference_params=state, num_last_tokens=1)
        if self.track_boundaries:
            self._prefill_length = input_ids.shape[1]
            self._prefill_chunks = [b.boundary_mask.sum() for b in output.bpred_output]
        return output.logits[:, -1]

    def decode_step(self, last_ids: torch.Tensor, state: Any) -> torch.Tensor:
        if last_ids.shape[0] != 1:
            raise NotImplementedError("HNetForCausalLM.step solo admite batch_size = 1.")
        output = self.model.step(last_ids, state)
        if self.track_boundaries:
            # La lista solo contiene las etapas que han avanzado en este paso
            for stage, bpred in enumerate(output.bpred_output):
                self._decode_chunks[stage] += bpred.boundary_mask.sum()
            self._decode_steps += 1
        return output.logits[:, -1]

    @torch.inference_mode()
    def segment(self, text: str) -> list[list[int]]:
        """
        Devuelve, por etapa, los desplazamientos en bytes de ``text`` donde comienza cada chunk.

        Se ejecuta un forward completo en modo empaquetado (sin caché), cuyas
        fronteras coinciden con las de la generación incremental. Las máscaras
        de la etapa ``s`` se expresan sobre los chunks de la etapa ``s - 1``,
        por lo que se componen para traducirlas a posiciones de byte. El byte
        BOS antepuesto se descarta.
        """
        input_ids = self.encode(text)
        output = self.model(input_ids, num_last_tokens=1)
        positions = torch.arange(input_ids.shape[1], device=self.device)
        offset = 1 if self.add_bos else 0
        stages = []
        for bpred in output.bpred_output:
            positions = positions[bpred.boundary_mask.flatten()]
            starts = positions[positions >= offset] - offset
            stages.append(starts.tolist())
        return stages

    def collect_extra(self, state: Any) -> dict:
        if not self.track_boundaries:
            return {}
        prefill_chunks = [int(c.item()) for c in self._prefill_chunks]
        decode_chunks = self._decode_chunks.tolist()
        return {
            "model_name": self.model_name,
            "prefill_units": self._prefill_length,
            "prefill_chunks_per_stage": prefill_chunks,
            "prefill_compression_per_stage": _compression(self._prefill_length, prefill_chunks),
            "decode_units": self._decode_steps,
            "decode_chunks_per_stage": decode_chunks,
            "decode_compression_per_stage": _compression(self._decode_steps, decode_chunks),
        }


def _compression(n_units: int, chunks_per_stage: list[int]) -> list[float | None]:
    """Tasa de compresión de cada etapa respecto a la anterior (unidades de entrada / chunks)."""
    ratios = []
    previous = n_units
    for chunks in chunks_per_stage:
        ratios.append(previous / chunks if chunks else None)
        previous = chunks
    return ratios
