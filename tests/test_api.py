"""
Pruebas del motor de inferencia (``src/core/engine.py``) y de la API REST (``src/api``).

Se emplea el modelo ficticio ``ScriptedWrapper`` de ``tests.test_engine`` en
CPU, por lo que no se requiere GPU ni checkpoints. Las pruebas de la API se
omiten si FastAPI no está instalado.

Ejecución:
    .venv/bin/python -m unittest tests.test_api -v
"""

import asyncio
import json
import os
import sys
import time
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.core.engine import (  # noqa: E402
    EngineConfig,
    EngineError,
    GenerateRequest,
    InferenceEngine,
    ModelNotAvailable,
    ModelSpec,
    _StreamDecoder,
    build_segmentation,
)
from tests.test_engine import ScriptedWrapper, _greedy  # noqa: E402

try:
    import fastapi  # noqa: F401
    from fastapi.testclient import TestClient

    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False

CATALOG = [ModelSpec("ficticio-a", "test"), ModelSpec("ficticio-b", "test")]
SCRIPT = [0, 1, 2]                          # "abc" de forma cíclica al agotarse: "abccc..."


class SlowScriptedWrapper(ScriptedWrapper):
    """Modelo ficticio con un retardo por paso, para observar la cancelación."""

    def decode_step(self, last_ids, state):
        time.sleep(0.005)
        return super().decode_step(last_ids, state)


def make_engine(wrapper_cls=ScriptedWrapper, **config) -> InferenceEngine:
    config.setdefault("default_model", "ficticio-a")
    config.setdefault("warmup_iterations", 0)

    def factory(spec: ModelSpec):
        wrapper = wrapper_cls(SCRIPT)
        wrapper.load(spec.name)
        return wrapper

    return InferenceEngine(EngineConfig(device="cpu", **config), catalog=CATALOG, wrapper_factory=factory)


def run(coro):
    return asyncio.run(coro)


async def collect(events) -> list[dict]:
    return [event async for event in events]


# ======================================================================
# 1. Utilidades de texto
# ======================================================================


class TestSegmentation(unittest.TestCase):
    def test_boundaries_inside_multibyte_characters_are_snapped(self):
        seg = build_segmentation("añb", [[0, 1, 2, 3]], ["nivel"])
        level = seg.levels[0]
        self.assertEqual(level.segments, ["a", "ñ", "b"])
        self.assertEqual(level.n_units, 4)                      # unidades originales del modelo
        self.assertEqual(seg.n_bytes, 4)
        self.assertEqual("".join(level.segments), "añb")

    def test_first_segment_always_starts_at_zero(self):
        seg = build_segmentation("hola mundo", [[5]], ["nivel"])
        self.assertEqual(seg.levels[0].segments, ["hola ", "mundo"])

    def test_several_levels(self):
        seg = build_segmentation("abcdef", [[0, 2, 4], [0, 4]], ["etapa 1", "etapa 2"])
        self.assertEqual([lvl.segments for lvl in seg.levels], [["ab", "cd", "ef"], ["abcd", "ef"]])
        self.assertEqual(seg.levels[0].bytes_per_unit, 2.0)


class TestStreamDecoder(unittest.TestCase):
    def test_multibyte_character_split_across_units(self):
        decoder = _StreamDecoder(holdback=0)
        self.assertEqual(decoder.feed(b"\xc3"), "")
        self.assertEqual(decoder.feed(b"\xb1"), "ñ")

    def test_holdback_retains_last_bytes(self):
        decoder = _StreamDecoder(holdback=2)
        self.assertEqual(decoder.feed(b"abc"), "a")
        self.assertEqual(decoder.finish("abc"), "bc")

    def test_finish_with_excluded_stop_sequence(self):
        decoder = _StreamDecoder(holdback=2)
        decoder.feed(b"abc")
        self.assertEqual(decoder.finish("a"), "")


# ======================================================================
# 2. Motor
# ======================================================================


class TestEngine(unittest.TestCase):
    def test_start_loads_default_model(self):
        async def scenario():
            engine = make_engine()
            await engine.start()
            status = engine.status()
            engine.shutdown()
            return status, engine.available_models()

        status, models = run(scenario())
        self.assertEqual(status["model"]["model"], "ficticio-a")
        self.assertIsNone(status["gpu"])
        self.assertEqual([m.loaded for m in models], [True, False])

    def test_generate_report(self):
        async def scenario():
            engine = make_engine()
            await engine.start()
            report = await engine.generate(GenerateRequest("xy", _greedy(max_new_bytes=5), segment=True))
            served = engine.status()["served_requests"]
            engine.shutdown()
            return report, served

        report, served = run(scenario())
        self.assertEqual(report.text, "abccc")
        self.assertEqual(report.model, "ficticio-a")
        self.assertEqual(report.stop_reason, "max_bytes")
        self.assertEqual(report.n_new_bytes, 5)
        self.assertIsNone(report.peak_vram_mib)
        self.assertEqual(report.segmentation.text, "xyabccc")
        self.assertEqual(served, 1)

    def test_stream_deltas_reconstruct_text(self):
        async def scenario():
            engine = make_engine()
            await engine.start()
            events = await collect(engine.stream(GenerateRequest("xy", _greedy(max_new_bytes=6))))
            engine.shutdown()
            return events

        events = run(scenario())
        kinds = [e["event"] for e in events]
        self.assertEqual(kinds[0], "start")
        self.assertEqual(kinds[-1], "done")
        streamed = "".join(e["text"] for e in events if e["event"] == "delta")
        self.assertEqual(streamed, events[-1]["report"]["text"])

    def test_stream_never_emits_stop_sequence(self):
        async def scenario():
            engine = make_engine()
            await engine.start()
            config = _greedy(max_new_bytes=10, stop_sequences=["bc"])
            events = await collect(engine.stream(GenerateRequest("xy", config)))
            engine.shutdown()
            return events

        events = run(scenario())
        deltas = [e["text"] for e in events if e["event"] == "delta"]
        self.assertEqual("".join(deltas), "a")
        self.assertEqual(events[-1]["report"]["stop_reason"], "stop_sequence")

    def test_stream_switches_model_on_demand(self):
        async def scenario():
            engine = make_engine()
            await engine.start()
            events = await collect(engine.stream(GenerateRequest("xy", _greedy(max_new_bytes=2), model="ficticio-b")))
            loaded = engine.status()["model"]["model"]
            engine.shutdown()
            return events, loaded

        events, loaded = run(scenario())
        self.assertEqual(events[0], {"event": "status", "message": "Cargando el modelo ficticio-b..."})
        self.assertEqual(events[-1]["report"]["model"], "ficticio-b")
        self.assertEqual(loaded, "ficticio-b")

    def test_abandoned_stream_cancels_generation(self):
        async def scenario():
            engine = make_engine(wrapper_cls=SlowScriptedWrapper, max_new_bytes=2048)
            await engine.start()
            stream = engine.stream(GenerateRequest("xy", _greedy(max_new_bytes=2000)))
            async for event in stream:
                if event["event"] == "delta":
                    break
            await stream.aclose()
            await engine._submit(lambda: None)            # espera a que el hilo de la GPU quede libre
            steps = engine._wrapper.decode_calls
            engine.shutdown()
            return steps

        self.assertLess(run(scenario()), 100)

    def test_compare_runs_models_in_sequence(self):
        async def scenario():
            engine = make_engine()
            await engine.start()
            events = await collect(engine.compare("xy", ["ficticio-b", "ficticio-a"], _greedy(max_new_bytes=3)))
            loaded = engine.status()["model"]["model"]
            engine.shutdown()
            return events, loaded

        events, loaded = run(scenario())
        done = [e for e in events if e["event"] == "done"]
        self.assertEqual([e["report"]["model"] for e in done], ["ficticio-b", "ficticio-a"])
        self.assertEqual([e["index"] for e in done], [0, 1])
        self.assertTrue(all(e["report"]["segmentation"] for e in done))
        self.assertEqual(events[-1], {"event": "end"})
        self.assertEqual(loaded, "ficticio-a")

    def test_validation_errors(self):
        async def scenario():
            engine = make_engine(max_new_bytes=16, max_prompt_bytes=8)
            await engine.start()
            errors = []
            for request in (
                GenerateRequest("", _greedy(max_new_bytes=4)),
                GenerateRequest("x" * 9, _greedy(max_new_bytes=4)),
                GenerateRequest("x", _greedy(max_new_bytes=17)),
            ):
                with self.assertRaises(EngineError) as ctx:
                    await engine.generate(request)
                errors.append(ctx.exception)
            with self.assertRaises(ModelNotAvailable):
                await engine.load_model("inexistente")
            engine.shutdown()

        run(scenario())


# ======================================================================
# 3. API REST
# ======================================================================


def parse_sse(text: str) -> list[dict]:
    events = []
    for block in text.strip().split("\n\n"):
        lines = dict(line.split(": ", 1) for line in block.split("\n"))
        events.append({"event": lines["event"], **json.loads(lines["data"])})
    return events


@unittest.skipUnless(HAS_FASTAPI, "Se requiere FastAPI.")
class TestAPI(unittest.TestCase):
    def setUp(self):
        from src.api.app import create_app

        self.client_cm = TestClient(create_app(make_engine()))
        self.client = self.client_cm.__enter__()

    def tearDown(self):
        self.client_cm.__exit__(None, None, None)

    def test_health_status_and_models(self):
        self.assertEqual(self.client.get("/health").json(), {"status": "ok"})
        self.assertEqual(self.client.get("/status").json()["model"]["model"], "ficticio-a")
        models = self.client.get("/models").json()
        self.assertEqual([m["name"] for m in models], ["ficticio-a", "ficticio-b"])

    def test_index_page(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Motor de inferencia H-Net", response.text)

    def test_generate(self):
        body = {"prompt": "xy", "max_new_bytes": 4, "temperature": 0.0, "segment": True}
        report = self.client.post("/generate", json=body).json()
        self.assertEqual(report["text"], "abcc")
        self.assertEqual(report["segmentation"]["levels"][0]["name"], "Tokens BPE")

    def test_generate_stream(self):
        body = {"prompt": "xy", "max_new_bytes": 4, "temperature": 0.0}
        response = self.client.post("/generate/stream", json=body)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        events = parse_sse(response.text)
        streamed = "".join(e["text"] for e in events if e["event"] == "delta")
        self.assertEqual(streamed, events[-1]["report"]["text"])

    def test_compare_stream(self):
        body = {"prompt": "xy", "models": ["ficticio-a", "ficticio-b"], "max_new_bytes": 3, "temperature": 0.0}
        events = parse_sse(self.client.post("/compare/stream", json=body).text)
        self.assertEqual([e["report"]["model"] for e in events if e["event"] == "done"], ["ficticio-a", "ficticio-b"])

    def test_segment(self):
        seg = self.client.post("/segment", json={"text": "hola"}).json()
        self.assertEqual("".join(seg["levels"][0]["segments"]), "hola")

    def test_load_model(self):
        self.assertEqual(self.client.post("/models/load", json={"name": "ficticio-b"}).json()["model"], "ficticio-b")

    def test_errors(self):
        self.assertEqual(self.client.post("/models/load", json={"name": "otro"}).status_code, 404)
        self.assertEqual(
            self.client.post("/generate/stream", json={"prompt": "x", "model": "otro"}).status_code, 404
        )
        self.assertEqual(self.client.post("/generate", json={"prompt": "x", "max_new_bytes": 10**6}).status_code, 400)
        self.assertEqual(self.client.post("/generate", json={"prompt": "x", "temperature": -1}).status_code, 422)
        self.assertEqual(self.client.post("/generate", json={"prompt": ""}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
