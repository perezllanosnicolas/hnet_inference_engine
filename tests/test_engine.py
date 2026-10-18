"""
Pruebas del motor de inferencia.

Se agrupan en tres bloques:

1. Coherencia de la decodificación incremental de H-Net: los logits obtenidos
   mediante prefill + ``step`` (con forzado del byte real en cada paso) deben
   coincidir, dentro de la tolerancia de bf16, con los del forward completo sin
   caché, y las decisiones de frontera del routing deben ser idénticas. Se
   verifica con un modelo pequeño de pesos aleatorios y, si está disponible en la
   caché local de Hugging Face, con el checkpoint ``hnet_1stage_L``. Requieren GPU.

2. Baseline Qwen (``QwenWrapper``): la tabla de bytes reproduce el UTF-8 del
   texto y la generación voraz coincide con ``model.generate`` de
   ``transformers``, tanto con caché dinámica como estática, y las tablas de
   RoPE compactas coinciden bit a bit con las originales sin alterar la
   generación. Requieren GPU y
   ``Qwen/Qwen2.5-0.5B`` en la caché local de Hugging Face.

3. Condiciones de parada y muestreo de ``BaseModelWrapper.generate`` mediante un
   modelo ficticio en CPU cuyos logits están predeterminados, de modo que el
   comportamiento esperado es exacto y no depende de pesos reales.

Ejecución: ``.venv/bin/python -m unittest tests.test_engine -v`` desde la raíz
del repositorio (también compatible con pytest).
"""

import os
import sys
import unittest

import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.models.wrappers.base import BaseModelWrapper, GenerationConfig  # noqa: E402

HAS_CUDA = torch.cuda.is_available()
PROMPT = "Hola URJC, se comprueba la decodificación incremental de H-Net."
PREFILL_LEN = 16


# ======================================================================
# 1. Coherencia prefill + step frente a forward completo (H-Net, GPU)
# ======================================================================


def _flat_boundaries(bpred_output, stage: int = 0) -> torch.Tensor:
    return bpred_output[stage].boundary_mask.reshape(-1)


def _teacher_forced_logits(model, input_ids: torch.Tensor, prefill_len: int, dtype):
    """
    Devuelve los logits ``(L, V)`` y las fronteras de la etapa 0 ``(L,)`` obtenidos
    por prefill de los ``prefill_len`` primeros bytes seguido de un ``step`` por
    cada byte restante.
    """
    length = input_ids.shape[1]
    cache = model.allocate_inference_cache(1, length, dtype=dtype)
    prompt = input_ids[:, :prefill_len]
    out = model(prompt, mask=torch.ones_like(prompt, dtype=torch.bool), inference_params=cache)
    logits = [out.logits[0].float()]
    boundaries = [_flat_boundaries(out.bpred_output)]
    for t in range(prefill_len, length):
        out = model.step(input_ids[:, t : t + 1], cache)
        logits.append(out.logits[0].float())
        boundaries.append(_flat_boundaries(out.bpred_output))
    return torch.cat(logits, dim=0), torch.cat(boundaries, dim=0)


def _full_forward(model, input_ids: torch.Tensor):
    """Forward completo sin caché: logits ``(L, V)`` y fronteras de la etapa 0 ``(L,)``."""
    out = model(input_ids, mask=torch.ones_like(input_ids, dtype=torch.bool))
    return out.logits[0].float(), _flat_boundaries(out.bpred_output)


class _CoherenceAssertions:
    """Comprobaciones comunes de coherencia entre ambas rutas de inferencia."""

    # Error absoluto máximo admitido en bf16, relativo a la escala de los logits
    rel_tol = 0.02

    def assert_coherent(self, model, input_ids: torch.Tensor, dtype):
        with torch.inference_mode():
            full_logits, full_bounds = _full_forward(model, input_ids)
            inc_logits, inc_bounds = _teacher_forced_logits(model, input_ids, PREFILL_LEN, dtype)

        self.assertEqual(inc_logits.shape, full_logits.shape)
        self.assertTrue(
            torch.equal(inc_bounds.bool(), full_bounds.bool()),
            "Las fronteras de chunk de la ruta incremental difieren de las del forward completo.",
        )

        max_err = (inc_logits - full_logits).abs().max().item()
        scale = full_logits.abs().max().item()
        self.assertLessEqual(
            max_err,
            self.rel_tol * scale,
            f"Error máximo {max_err:.4f} superior al {self.rel_tol:.0%} de la escala ({scale:.3f}).",
        )

        # El argmax solo se exige donde el margen top-1/top-2 no puede invertirse por el error
        # numérico observado; con pesos aleatorios los logits presentan empates aproximados.
        top2 = full_logits.topk(2, dim=-1).values
        decisive = (top2[:, 0] - top2[:, 1]) > 2 * max_err
        self.assertGreater(decisive.sum().item(), 0)
        agree = inc_logits.argmax(-1) == full_logits.argmax(-1)
        self.assertTrue(
            agree[decisive].all(),
            f"El argmax difiere en {(~agree[decisive]).sum().item()} posiciones con margen suficiente.",
        )


@unittest.skipUnless(HAS_CUDA, "Se requiere GPU (FlashAttention y kernels de Mamba).")
class TestHNetIncrementalRandom(_CoherenceAssertions, unittest.TestCase):
    """Coherencia sobre un H-Net pequeño de pesos aleatorios (configuración del smoke test)."""

    @classmethod
    def setUpClass(cls):
        from src.models.architectures import hnet_compat  # noqa: F401
        from hnet.models.config_hnet import AttnConfig, HNetConfig, SSMConfig
        from hnet.models.mixer_seq import HNetForCausalLM

        cls.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        torch.manual_seed(0)
        config = HNetConfig(
            vocab_size=256,
            d_model=[128, 256, 128],
            d_intermediate=[256, 512, 256],
            arch_layout=["m1t1", ["M2T1"], "m1t1"],
            attn_cfg=AttnConfig(
                num_heads=[4, 8, 4], rotary_emb_dim=[0, 0, 0], window_size=[-1, -1, -1]
            ),
            ssm_cfg=SSMConfig(),
        )
        cls.model = HNetForCausalLM(config, device="cuda", dtype=cls.dtype).eval()
        data = bytes([254]) + PROMPT.encode("utf-8")
        cls.input_ids = torch.tensor([list(data)], dtype=torch.long, device="cuda")

    @classmethod
    def tearDownClass(cls):
        del cls.model
        torch.cuda.empty_cache()

    def test_prefill_step_matches_full_forward(self):
        self.assert_coherent(self.model, self.input_ids, self.dtype)

    def test_prefill_of_whole_sequence_matches_last_logits(self):
        """El prefill completo con ``num_last_tokens=1`` reproduce la última posición."""
        ids = self.input_ids
        with torch.inference_mode():
            full_logits, _ = _full_forward(self.model, ids)
            cache = self.model.allocate_inference_cache(1, ids.shape[1], dtype=self.dtype)
            out = self.model(
                ids,
                mask=torch.ones_like(ids, dtype=torch.bool),
                inference_params=cache,
                num_last_tokens=1,
            )
        self.assertEqual(tuple(out.logits.shape), (1, 1, 256))
        last = out.logits[0, -1].float()
        torch.testing.assert_close(last, full_logits[-1], rtol=0.0, atol=1e-2 * full_logits.abs().max().item())


def _cached_checkpoint(name: str) -> str | None:
    """Ruta local del checkpoint en la caché de Hugging Face, sin acceso a red."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    path = try_to_load_from_cache(f"cartesia-ai/{name}", f"{name}.pt")
    return path if isinstance(path, str) and os.path.isfile(path) else None


CHECKPOINT_NAME = "hnet_1stage_L"
CHECKPOINT_PATH = _cached_checkpoint(CHECKPOINT_NAME) if HAS_CUDA else None


@unittest.skipUnless(CHECKPOINT_PATH, f"Se requiere GPU y '{CHECKPOINT_NAME}' en la caché local.")
class TestHNetIncrementalCheckpoint(_CoherenceAssertions, unittest.TestCase):
    """Coherencia sobre el checkpoint preentrenado, cargado mediante ``HNetWrapper``."""

    @classmethod
    def setUpClass(cls):
        from src.models.wrappers.hnet_wrapper import CONFIG_DIR, HNetWrapper

        cls.wrapper = HNetWrapper(device="cuda")
        cls.wrapper.load(CHECKPOINT_PATH, config=str(CONFIG_DIR / f"{CHECKPOINT_NAME}.json"))
        cls.dtype = cls.wrapper.dtype
        cls.input_ids = cls.wrapper.encode(PROMPT)

    @classmethod
    def tearDownClass(cls):
        del cls.wrapper
        torch.cuda.empty_cache()

    def test_prefill_step_matches_full_forward(self):
        self.assert_coherent(self.wrapper.model, self.input_ids, self.dtype)

    def test_wrapper_greedy_matches_full_forward(self):
        """La generación voraz del wrapper coincide con el argmax del forward completo."""
        config = GenerationConfig(max_new_bytes=24, temperature=0.0, ignore_eos=True, measure_time=False)
        result = self.wrapper.generate(PROMPT, config)
        self.assertEqual(result.n_new_units, 24)

        sequence = torch.cat([self.input_ids[0], result.output_ids.to(self.input_ids.device)])
        with torch.inference_mode():
            full_logits, _ = _full_forward(self.wrapper.model, sequence.unsqueeze(0))
        n_prompt = self.input_ids.shape[1]
        expected = full_logits[n_prompt - 1 : -1].argmax(-1).cpu()
        self.assertTrue(torch.equal(expected, result.output_ids))

        extra = result.extra
        self.assertEqual(extra["prefill_units"], n_prompt)
        self.assertEqual(extra["decode_units"], result.n_new_units - 1)
        self.assertGreater(extra["prefill_chunks_per_stage"][0], 0)


# ======================================================================
# 2. Baseline Qwen: contabilidad de bytes y generación voraz (GPU)
# ======================================================================

QWEN_REPO = "Qwen/Qwen2.5-0.5B"


def _qwen_cached(repo_id: str) -> bool:
    """Indica si los pesos del modelo están en la caché local de Hugging Face."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    path = try_to_load_from_cache(repo_id, "model.safetensors")
    return isinstance(path, str) and os.path.isfile(path)


QWEN_AVAILABLE = HAS_CUDA and _qwen_cached(QWEN_REPO)


class TestQwenWrapperValidation(unittest.TestCase):
    """Validación de argumentos sin cargar el modelo."""

    def test_static_cache_rejects_flash_attention(self):
        from src.models.wrappers.qwen_wrapper import QwenWrapper

        with self.assertRaises(ValueError):
            QwenWrapper(device="cpu", cache_implementation="static")

    def test_checkpoint_resolution(self):
        from src.models.wrappers.qwen_wrapper import resolve_checkpoint

        self.assertEqual(resolve_checkpoint("Qwen2.5-0.5B"), "Qwen/Qwen2.5-0.5B")
        self.assertEqual(resolve_checkpoint("Qwen/Qwen2.5-1.5B"), "Qwen/Qwen2.5-1.5B")


@unittest.skipUnless(QWEN_AVAILABLE, f"Se requiere GPU y '{QWEN_REPO}' en la caché local.")
class TestQwenWrapper(unittest.TestCase):
    """Coherencia de ``QwenWrapper`` con ``transformers`` sobre ``Qwen2.5-0.5B``."""

    @classmethod
    def setUpClass(cls):
        from src.models.wrappers.qwen_wrapper import QwenWrapper

        cls.wrapper = QwenWrapper(device="cuda")
        cls.wrapper.load(QWEN_REPO)
        cls.static = QwenWrapper(device="cuda", attn_implementation="sdpa", cache_implementation="static")
        cls.static.load(QWEN_REPO)
        cls.original = QwenWrapper(device="cuda", compact_rope=False)
        cls.original.load(QWEN_REPO)

    @classmethod
    def tearDownClass(cls):
        del cls.wrapper, cls.static, cls.original
        torch.cuda.empty_cache()

    def test_byte_table_reconstructs_text(self):
        """La concatenación de bytes por token reproduce el UTF-8 original, incluidos caracteres partidos."""
        for text in [PROMPT, "日本語のテキスト y emoji \U0001F600", "def f(x):\n\treturn x ** 2\n"]:
            ids = self.wrapper.encode(text)[0].tolist()
            data = b"".join(self.wrapper.unit_to_bytes(i) for i in ids)
            self.assertEqual(data, text.encode("utf-8"))

    def test_special_tokens_have_no_bytes(self):
        for token_id in self.wrapper.eos_unit_ids + self.wrapper.tokenizer.all_special_ids:
            self.assertEqual(self.wrapper.unit_to_bytes(token_id), b"")
        # Identificadores de relleno de la matriz de embeddings, sin token asociado
        self.assertEqual(self.wrapper.unit_to_bytes(self.wrapper.model.config.vocab_size - 1), b"")

    def _assert_matches_hf_generate(self, wrapper):
        config = GenerationConfig(max_new_bytes=64, temperature=0.0, ignore_eos=True, measure_time=False)
        result = wrapper.generate(PROMPT, config)
        input_ids = wrapper.encode(PROMPT)
        with torch.inference_mode():
            reference = wrapper.model.generate(
                input_ids,
                attention_mask=torch.ones_like(input_ids),
                max_new_tokens=result.n_new_units,
                do_sample=False,
                eos_token_id=None,
                pad_token_id=wrapper.eos_unit_ids[0],
            )
        expected = reference[0, input_ids.shape[1] :].cpu()
        self.assertTrue(torch.equal(expected, result.output_ids))
        self.assertGreaterEqual(result.n_new_bytes, 64)
        self.assertGreater(result.bytes_per_unit, 1.0)

    def test_greedy_matches_hf_generate(self):
        """Caché dinámica con FlashAttention: misma secuencia voraz que ``model.generate``."""
        self._assert_matches_hf_generate(self.wrapper)

    def test_static_cache_greedy_matches_hf_generate(self):
        """Caché estática con SDPA: misma secuencia voraz que ``model.generate`` (caché dinámica)."""
        self._assert_matches_hf_generate(self.static)

    def _assert_rope_matches_original(self, wrapper):
        reference = self.original.model.model.layers[0].self_attn.rotary_emb
        length = wrapper.rope_length
        for module in wrapper._rotary_modules():
            self.assertEqual(module.max_seq_len_cached, length)
            self.assertTrue(torch.equal(module.cos_cached, reference.cos_cached[:length]))
            self.assertTrue(torch.equal(module.sin_cached, reference.sin_cached[:length]))

    def test_compact_rope_tables_match_original(self):
        """Las tablas compactas coinciden bit a bit con el prefijo de las originales y se comparten."""
        self._assert_rope_matches_original(self.wrapper)
        pointers = {m.cos_cached.data_ptr() for m in self.wrapper._rotary_modules()}
        self.assertEqual(len(pointers), 1)
        self.assertIsNone(self.original.rope_length)
        self.assertEqual(self.original.model.model.layers[0].self_attn.rotary_emb.max_seq_len_cached,
                         self.original.model.config.max_position_embeddings)

    def test_compact_rope_tables_grow_on_demand(self):
        """``allocate_state`` amplía las tablas hasta un múltiplo de la granularidad."""
        from src.models.wrappers.qwen_wrapper import ROPE_GRANULARITY, QwenWrapper

        wrapper = QwenWrapper(device="cuda")
        wrapper.load(QWEN_REPO)
        self.assertEqual(wrapper.rope_length, ROPE_GRANULARITY)
        wrapper.allocate_state(batch_size=1, max_seqlen=ROPE_GRANULARITY + 1)
        self.assertEqual(wrapper.rope_length, 2 * ROPE_GRANULARITY)
        self._assert_rope_matches_original(wrapper)
        wrapper.allocate_state(batch_size=1, max_seqlen=8)
        self.assertEqual(wrapper.rope_length, 2 * ROPE_GRANULARITY)
        del wrapper
        torch.cuda.empty_cache()

    def test_compact_rope_preserves_generation(self):
        """La generación voraz es idéntica con tablas compactas y con las originales."""
        config = GenerationConfig(max_new_bytes=600, temperature=0.0, ignore_eos=True, measure_time=False)
        compact = self.wrapper.generate(PROMPT, config)
        original = self.original.generate(PROMPT, config)
        self.assertTrue(torch.equal(compact.output_ids, original.output_ids))
        self.assertGreater(self.wrapper.rope_length, 256)

    def test_static_cache_capacity_is_enforced(self):
        input_ids = self.static.encode(PROMPT)
        state = self.static.allocate_state(batch_size=1, max_seqlen=input_ids.shape[1] - 1)
        with torch.inference_mode(), self.assertRaises(RuntimeError):
            self.static.prefill(input_ids, state)


# ======================================================================
# 3. Parada y muestreo de BaseModelWrapper (modelo ficticio, CPU)
# ======================================================================

# Vocabulario ficticio: unidades de 1 byte, una de 2 bytes y dos especiales
VOCAB = {0: b"a", 1: b"b", 2: b"c", 3: b"\xc3\xb1", 4: b"", 5: b""}  # 3 = "ñ", 4 = EOS, 5 = especial
EOS_ID = 4
VOCAB_SIZE = len(VOCAB)


class ScriptedWrapper(BaseModelWrapper):
    """
    Modelo ficticio que emite una secuencia predeterminada de unidades.

    En el paso ``i`` devuelve logits con un margen ``margin`` a favor de
    ``script[i]`` (o de la última unidad si el guion se agota), de forma que la
    decodificación voraz reproduce el guion exactamente.
    """

    unit = "token"

    def __init__(self, script: list[int], margin: float = 10.0, logits_fn=None):
        super().__init__(device="cpu", dtype=torch.float32)
        self.script = script
        self.margin = margin
        self.logits_fn = logits_fn
        self.step_index = 0
        self.prefill_calls = 0
        self.decode_calls = 0

    def load(self, checkpoint: str, config: str | None = None) -> None:
        self.model = torch.nn.Identity()

    def encode(self, text: str) -> torch.Tensor:
        ids = [b % 3 for b in text.encode("utf-8")]  # proyección arbitraria sobre {a, b, c}
        return torch.tensor([ids], dtype=torch.long)

    def unit_to_bytes(self, unit_id: int) -> bytes:
        return VOCAB[unit_id]

    @property
    def eos_unit_ids(self) -> list[int]:
        return [EOS_ID]

    def allocate_state(self, batch_size: int, max_seqlen: int):
        self.step_index = 0
        return {"max_seqlen": max_seqlen}

    def _next_logits(self) -> torch.Tensor:
        if self.logits_fn is not None:
            logits = self.logits_fn(self.step_index)
        else:
            target = self.script[min(self.step_index, len(self.script) - 1)]
            logits = torch.zeros(1, VOCAB_SIZE)
            logits[0, target] = self.margin
        self.step_index += 1
        return logits

    def prefill(self, input_ids: torch.Tensor, state) -> torch.Tensor:
        self.prefill_calls += 1
        return self._next_logits()

    def decode_step(self, last_ids: torch.Tensor, state) -> torch.Tensor:
        self.decode_calls += 1
        return self._next_logits()


def _greedy(**kwargs) -> GenerationConfig:
    kwargs.setdefault("temperature", 0.0)
    kwargs.setdefault("measure_time", False)
    return GenerationConfig(**kwargs)


def _loaded(script, **kwargs) -> ScriptedWrapper:
    wrapper = ScriptedWrapper(script, **kwargs)
    wrapper.load("ficticio")
    return wrapper


class TestStopConditions(unittest.TestCase):
    def test_stops_at_max_new_bytes(self):
        wrapper = _loaded([0, 1, 2, 0, 1, 2])
        result = wrapper.generate("xy", _greedy(max_new_bytes=4))
        self.assertEqual(result.stop_reason, "max_bytes")
        self.assertEqual(result.text, "abca")
        self.assertEqual(result.output_ids.tolist(), [0, 1, 2, 0])
        self.assertEqual((result.n_new_units, result.n_new_bytes), (4, 4))
        # La última unidad no requiere un paso de decodificación adicional
        self.assertEqual((wrapper.prefill_calls, wrapper.decode_calls), (1, 3))

    def test_multibyte_unit_may_exceed_limit(self):
        wrapper = _loaded([0, 3, 0])
        result = wrapper.generate("x", _greedy(max_new_bytes=2))
        self.assertEqual(result.stop_reason, "max_bytes")
        self.assertEqual(result.text, "añ")
        self.assertEqual((result.n_new_units, result.n_new_bytes), (2, 3))
        self.assertAlmostEqual(result.bytes_per_unit, 1.5)

    def test_stops_at_eos_without_emitting_bytes(self):
        wrapper = _loaded([0, 1, EOS_ID, 2])
        result = wrapper.generate("x", _greedy(max_new_bytes=10))
        self.assertEqual(result.stop_reason, "eos")
        self.assertEqual(result.text, "ab")
        self.assertEqual(result.output_ids.tolist(), [0, 1, EOS_ID])
        self.assertEqual((result.n_new_units, result.n_new_bytes), (3, 2))

    def test_ignore_eos_reaches_max_bytes(self):
        wrapper = _loaded([0, EOS_ID, 1, 2])
        result = wrapper.generate("x", _greedy(max_new_bytes=3, ignore_eos=True))
        self.assertEqual(result.stop_reason, "max_bytes")
        self.assertEqual(result.text, "abc")
        self.assertEqual(result.output_ids.tolist(), [0, EOS_ID, 1, 2])

    def test_stops_at_stop_unit(self):
        wrapper = _loaded([0, 5, 1])
        result = wrapper.generate("x", _greedy(max_new_bytes=10, stop_unit_ids=[5]))
        self.assertEqual(result.stop_reason, "stop_unit")
        self.assertEqual(result.text, "a")
        self.assertEqual(result.output_ids.tolist(), [0, 5])

    def test_stop_unit_takes_effect_even_with_ignore_eos(self):
        wrapper = _loaded([0, EOS_ID, 1])
        config = _greedy(max_new_bytes=10, ignore_eos=True, stop_unit_ids=[EOS_ID])
        self.assertEqual(wrapper.generate("x", config).stop_reason, "stop_unit")

    def test_stop_sequence_excluded_from_text_but_counted(self):
        wrapper = _loaded([0, 1, 2, 0, 1])
        result = wrapper.generate("x", _greedy(max_new_bytes=10, stop_sequences=["bc"]))
        self.assertEqual(result.stop_reason, "stop_sequence")
        self.assertEqual(result.text, "a")
        self.assertEqual(result.n_new_bytes, 3)
        self.assertEqual(result.n_new_units, 3)

    def test_stop_sequence_spanning_multibyte_unit(self):
        wrapper = _loaded([0, 3, 1])
        result = wrapper.generate("x", _greedy(max_new_bytes=10, stop_sequences=["ñ"]))
        self.assertEqual(result.stop_reason, "stop_sequence")
        self.assertEqual(result.text, "a")
        self.assertEqual(result.n_new_bytes, 3)

    def test_first_matching_condition_wins(self):
        """Si la secuencia de parada se completa en la unidad que agota el límite, prevalece la parada."""
        wrapper = _loaded([0, 1])
        result = wrapper.generate("x", _greedy(max_new_bytes=2, stop_sequences=["ab"]))
        self.assertEqual(result.stop_reason, "stop_sequence")
        self.assertEqual(result.text, "")

    def test_empty_stop_sequence_is_ignored(self):
        wrapper = _loaded([0, 1, 2])
        result = wrapper.generate("x", _greedy(max_new_bytes=3, stop_sequences=[""]))
        self.assertEqual(result.stop_reason, "max_bytes")
        self.assertEqual(result.text, "abc")

    def test_prompt_accounting(self):
        wrapper = _loaded([0])
        result = wrapper.generate("año", _greedy(max_new_bytes=1))
        self.assertEqual(result.n_prompt_bytes, 4)
        self.assertEqual(result.n_prompt_units, 4)
        self.assertEqual(result.unit, "token")

    def test_state_bound_covers_prompt_and_generation(self):
        captured = {}
        wrapper = _loaded([0])
        original = wrapper.allocate_state

        def spy(batch_size, max_seqlen):
            captured.update(batch_size=batch_size, max_seqlen=max_seqlen)
            return original(batch_size, max_seqlen)

        wrapper.allocate_state = spy
        wrapper.generate("xyz", _greedy(max_new_bytes=7))
        self.assertEqual(captured, {"batch_size": 1, "max_seqlen": 3 + 7})


class TestGenerateValidation(unittest.TestCase):
    def test_requires_loaded_model(self):
        wrapper = ScriptedWrapper([0])
        with self.assertRaises(RuntimeError):
            wrapper.generate("x", _greedy(max_new_bytes=1))

    def test_rejects_non_positive_limit(self):
        wrapper = _loaded([0])
        with self.assertRaises(ValueError):
            wrapper.generate("x", _greedy(max_new_bytes=0))

    def test_timing_disabled(self):
        result = _loaded([0, 1]).generate("x", _greedy(max_new_bytes=2, measure_time=False))
        self.assertIsNone(result.prefill_time_s)
        self.assertIsNone(result.decode_time_s)
        self.assertIsNone(result.decode_bytes_per_s)

    def test_timing_enabled_on_cpu(self):
        result = _loaded([0, 1, 2]).generate("x", _greedy(max_new_bytes=3, measure_time=True))
        self.assertGreaterEqual(result.prefill_time_s, 0.0)
        self.assertGreaterEqual(result.decode_time_s, 0.0)


class TestSampling(unittest.TestCase):
    logits = torch.tensor([[2.0, 1.0, 0.5, -1.0, -3.0, -3.0]])

    def test_greedy_is_argmax(self):
        ids = BaseModelWrapper.sample(self.logits, _greedy(max_new_bytes=1))
        self.assertEqual(ids.tolist(), [[0]])

    def test_top_k_one_is_argmax(self):
        config = GenerationConfig(max_new_bytes=1, temperature=5.0, top_k=1)
        gen = torch.Generator().manual_seed(0)
        draws = {BaseModelWrapper.sample(self.logits, config, gen).item() for _ in range(50)}
        self.assertEqual(draws, {0})

    def test_top_k_restricts_support(self):
        config = GenerationConfig(max_new_bytes=1, temperature=10.0, top_k=2)
        gen = torch.Generator().manual_seed(0)
        draws = {BaseModelWrapper.sample(self.logits, config, gen).item() for _ in range(300)}
        self.assertEqual(draws, {0, 1})

    def test_top_p_restricts_support(self):
        # Probabilidades ~ [0.6, 0.3, 0.1, 0, ...]: con top_p=0.7 sobreviven las dos primeras
        logits = torch.log(torch.tensor([[0.6, 0.3, 0.1, 1e-9, 1e-9, 1e-9]]))
        config = GenerationConfig(max_new_bytes=1, temperature=1.0, top_p=0.7)
        gen = torch.Generator().manual_seed(0)
        draws = {BaseModelWrapper.sample(logits, config, gen).item() for _ in range(300)}
        self.assertEqual(draws, {0, 1})

    def test_top_p_always_keeps_most_probable(self):
        config = GenerationConfig(max_new_bytes=1, temperature=1.0, top_p=1e-6)
        gen = torch.Generator().manual_seed(0)
        draws = {BaseModelWrapper.sample(self.logits, config, gen).item() for _ in range(50)}
        self.assertEqual(draws, {0})

    def test_sampling_frequencies_follow_distribution(self):
        probs = torch.tensor([0.5, 0.3, 0.2, 0.0, 0.0, 0.0])
        logits = torch.log(probs.clamp_min(1e-12)).expand(20000, -1)
        config = GenerationConfig(max_new_bytes=1, temperature=1.0)
        gen = torch.Generator().manual_seed(0)
        draws = BaseModelWrapper.sample(logits, config, gen).squeeze(-1)
        freqs = torch.bincount(draws, minlength=VOCAB_SIZE).float() / draws.numel()
        torch.testing.assert_close(freqs, probs, rtol=0.0, atol=0.02)

    def test_seed_reproducibility(self):
        uniform = lambda _: torch.zeros(1, VOCAB_SIZE)  # noqa: E731
        config = GenerationConfig(max_new_bytes=20, temperature=1.0, ignore_eos=True, seed=123, measure_time=False)
        first = _loaded([], logits_fn=uniform).generate("x", config)
        second = _loaded([], logits_fn=uniform).generate("x", config)
        self.assertTrue(torch.equal(first.output_ids, second.output_ids))
        self.assertGreater(len(set(first.output_ids.tolist())), 1)

    def test_repetition_penalty_formula(self):
        logits = torch.tensor([[2.0, -2.0, 1.0, -1.0]])
        seen = torch.tensor([[True, True, False, False]])
        out = BaseModelWrapper.apply_repetition_penalty(logits, seen, 2.0)
        self.assertEqual(out.tolist(), [[1.0, -4.0, 1.0, -1.0]])

    def test_repetition_penalty_changes_greedy_choice(self):
        # Sin penalización se repite siempre "a"; con ella se alterna con "b".
        # El prompt "b" se proyecta sobre la unidad 2 ("c"), ajena a la elección.
        logits_fn = lambda _: torch.tensor([[2.0, 1.5, 0.0, -5.0, -5.0, -5.0]])  # noqa: E731
        base = _greedy(max_new_bytes=3, ignore_eos=True)
        plain = _loaded([], logits_fn=logits_fn).generate("b", base)
        self.assertEqual(plain.text, "aaa")

        penalized_config = _greedy(max_new_bytes=3, ignore_eos=True, repetition_penalty=2.0)
        penalized = _loaded([], logits_fn=logits_fn).generate("b", penalized_config)
        self.assertEqual(penalized.text, "aba")

    def test_repetition_penalty_includes_prompt(self):
        # El prompt "a" se proyecta sobre la unidad 1 ("b"), que queda penalizada desde el inicio
        logits_fn = lambda _: torch.tensor([[1.0, 1.5, 0.0, -5.0, -5.0, -5.0]])  # noqa: E731
        config = _greedy(max_new_bytes=1, repetition_penalty=2.0)
        self.assertEqual(ScriptedWrapper([]).encode("a").tolist(), [[1]])
        result = _loaded([], logits_fn=logits_fn).generate("a", config)
        self.assertEqual(result.text, "a")


if __name__ == "__main__":
    unittest.main()
