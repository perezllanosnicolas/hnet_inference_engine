"""
Envoltorio del baseline Transformer denso (familia Qwen2 / Qwen2.5) para el motor.

Se emplea ``transformers`` 4.44.2, que da soporte a la arquitectura ``qwen2``
(Qwen2 y Qwen2.5) pero no a Qwen3. La atención por defecto es
``flash_attention_2``, de modo que el baseline utiliza los mismos kernels de
FlashAttention que la atención de H-Net.

Diferencias respecto a la llamada directa a ``Qwen2ForCausalLM``:

- El prefill proyecta sobre el vocabulario únicamente la última posición. La
  implementación de ``transformers`` 4.44.2 calcula y convierte a float32 los
  logits de toda la secuencia (``L x 151936``), lo que inflaría el pico de VRAM
  frente a H-Net, que emplea ``num_last_tokens=1``.
- La posición en la caché se lleva en un contador de CPU y se pasa como
  ``cache_position``, evitando que el modelo la infiera a partir de la caché.

Contabilidad de bytes: cada token BPE a nivel de byte se traduce a los bytes
UTF-8 exactos que representa mediante una tabla precalculada. Un carácter
multibyte puede quedar repartido entre varios tokens; la tabla conserva esos
fragmentos, por lo que la concatenación de bytes reproduce el texto original.
"""

import os
from dataclasses import dataclass
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache, StaticCache
from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

from src.models.wrappers.base import BaseModelWrapper

HF_ORGANIZATION = "Qwen"


def resolve_checkpoint(checkpoint: str) -> str:
    """
    Resuelve un checkpoint a un directorio local o a un identificador de Hugging Face.

    Se admite un directorio local, un identificador completo (``Qwen/Qwen2.5-0.5B``)
    o únicamente el nombre del modelo (``Qwen2.5-0.5B``), al que se antepone la
    organización ``Qwen``.
    """
    if os.path.isdir(checkpoint) or "/" in checkpoint:
        return checkpoint
    return f"{HF_ORGANIZATION}/{checkpoint}"


def build_byte_table(tokenizer, vocab_size: int) -> list[bytes]:
    """
    Construye la correspondencia identificador de token -> bytes UTF-8.

    Los tokens del vocabulario BPE se representan con el alfabeto de
    ``bytes_to_unicode`` (GPT-2) y se invierten a sus bytes originales. Los
    tokens añadidos no especiales (p. ej. ``<tool_call>``) se traducen a su
    contenido literal, igual que en ``tokenizer.decode``; los especiales y los
    identificadores de relleno de la matriz de embeddings (``>= len(tokenizer)``)
    no representan bytes.
    """
    byte_decoder = {char: byte for byte, char in bytes_to_unicode().items()}
    added = tokenizer.added_tokens_decoder
    table = [b""] * vocab_size
    for token_id, token in enumerate(tokenizer.convert_ids_to_tokens(range(len(tokenizer)))):
        if token_id >= vocab_size:
            break
        if token_id in added:
            table[token_id] = b"" if added[token_id].special else added[token_id].content.encode("utf-8")
        else:
            table[token_id] = bytes(byte_decoder[char] for char in token)
    return table


@dataclass
class QwenState:
    """Estado de inferencia: caché KV y posición de escritura de la siguiente unidad."""

    cache: DynamicCache | StaticCache
    position: int = 0
    capacity: int | None = None       # longitud máxima en caché estática; None en dinámica


class QwenWrapper(BaseModelWrapper):
    """
    Envoltorio de ``Qwen2ForCausalLM``.

    ``cache_implementation`` selecciona la caché KV:

    - ``"dynamic"``: ``DynamicCache``, que crece concatenando en cada paso. Es la
      opción por defecto de ``transformers`` y es compatible con
      ``flash_attention_2``.
    - ``"static"``: ``StaticCache``, preasignada para ``max_seqlen`` unidades, del
      mismo modo que ``allocate_inference_cache`` en H-Net. En 4.44.2
      ``flash_attention_2`` no enmascara las posiciones vacías de una caché
      estática, por lo que esta opción exige ``attn_implementation`` igual a
      ``"sdpa"`` o ``"eager"``.
    """

    unit = "token"

    def __init__(
        self,
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
        attn_implementation: Literal["flash_attention_2", "sdpa", "eager"] = "flash_attention_2",
        cache_implementation: Literal["dynamic", "static"] = "dynamic",
    ):
        super().__init__(device=device, dtype=dtype)
        if cache_implementation not in ("dynamic", "static"):
            raise ValueError(f"cache_implementation no válido: {cache_implementation!r}.")
        if cache_implementation == "static" and attn_implementation == "flash_attention_2":
            raise ValueError(
                "La caché estática no es compatible con flash_attention_2 en transformers 4.44.2; "
                "debe emplearse attn_implementation='sdpa' o 'eager'."
            )
        self.attn_implementation = attn_implementation
        self.cache_implementation = cache_implementation
        self.model_name: str | None = None
        self.tokenizer = None
        self._byte_table: list[bytes] = []
        self._eos_ids: list[int] = []

    def load(self, checkpoint: str, config: str | None = None) -> None:
        """
        Carga el modelo y su tokenizador.

        ``config`` no se utiliza: la configuración acompaña al checkpoint en
        formato de Hugging Face. Se mantiene por compatibilidad con la interfaz.
        """
        repo_id = resolve_checkpoint(checkpoint)
        self.model_name = repo_id.rstrip("/").split("/")[-1]
        self.tokenizer = AutoTokenizer.from_pretrained(repo_id)

        model = AutoModelForCausalLM.from_pretrained(
            repo_id, torch_dtype=self.dtype, attn_implementation=self.attn_implementation
        )
        self.model = model.to(self.device).eval()

        self._byte_table = build_byte_table(self.tokenizer, self.model.config.vocab_size)

        eos_ids = {self.tokenizer.eos_token_id}
        generation_eos = self.model.generation_config.eos_token_id
        if isinstance(generation_eos, int):
            eos_ids.add(generation_eos)
        elif generation_eos is not None:
            eos_ids.update(generation_eos)
        self._eos_ids = sorted(i for i in eos_ids if i is not None)

    def encode(self, text: str) -> torch.Tensor:
        ids = self.tokenizer.encode(text)
        if not ids:
            raise ValueError("El prompt no produce ningún token: Qwen no antepone BOS y requiere texto no vacío.")
        return torch.tensor(ids, dtype=torch.long, device=self.device).unsqueeze(0)

    def unit_to_bytes(self, unit_id: int) -> bytes:
        return self._byte_table[unit_id]

    @property
    def eos_unit_ids(self) -> list[int]:
        return self._eos_ids

    def allocate_state(self, batch_size: int, max_seqlen: int) -> Any:
        if self.cache_implementation == "static":
            cache = StaticCache(
                self.model.config,
                max_batch_size=batch_size,
                max_cache_len=max_seqlen,
                device=self.device,
                dtype=self.dtype,
            )
            return QwenState(cache=cache, capacity=max_seqlen)
        return QwenState(cache=DynamicCache())

    def _forward(self, input_ids: torch.Tensor, state: QwenState) -> torch.Tensor:
        """Avanza ``input_ids`` sobre la caché y devuelve los logits ``(B, V)`` de la última posición."""
        length = input_ids.shape[1]
        if state.capacity is not None and state.position + length > state.capacity:
            raise RuntimeError(
                f"Se excede la capacidad de la caché estática ({state.capacity} posiciones); "
                "las unidades especiales no consumen bytes y no se contemplan en max_seqlen."
            )
        cache_position = torch.arange(state.position, state.position + length, device=self.device)
        hidden = self.model.model(
            input_ids=input_ids,
            past_key_values=state.cache,
            use_cache=True,
            cache_position=cache_position,
        ).last_hidden_state
        state.position += length
        return self.model.lm_head(hidden[:, -1])

    def prefill(self, input_ids: torch.Tensor, state: Any) -> torch.Tensor:
        return self._forward(input_ids, state)

    def decode_step(self, last_ids: torch.Tensor, state: Any) -> torch.Tensor:
        return self._forward(last_ids, state)

    def collect_extra(self, state: Any) -> dict:
        return {
            "model_name": self.model_name,
            "attn_implementation": self.attn_implementation,
            "cache_implementation": self.cache_implementation,
        }
