"""
API REST del motor de inferencia y página web de demostración.

Endpoints:

- ``GET  /``                  página de demostración (``static/index.html``).
- ``GET  /health``            comprobación de disponibilidad.
- ``GET  /status``            modelo cargado, cola de peticiones y memoria de GPU.
- ``GET  /models``            catálogo de modelos y disponibilidad local.
- ``POST /models/load``       carga un modelo (sustituye al residente).
- ``POST /generate``          generación completa con métricas.
- ``POST /generate/stream``   generación en streaming (Server-Sent Events).
- ``POST /compare/stream``    mismo prompt sobre varios modelos, en secuencia (SSE).
- ``POST /segment``           segmentación de un texto en chunks (H-Net) o tokens (Qwen).

Ejecución:
    .venv/bin/python -m src.api.app --model hnet_1stage_L --port 8000
"""

import argparse
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from src.api import schemas
from src.core.engine import (
    EngineConfig,
    EngineError,
    GenerateRequest,
    InferenceEngine,
    ModelNotAvailable,
    to_sse,
)

STATIC_DIR = Path(__file__).resolve().parent / "static"
SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def create_app(engine: InferenceEngine | None = None) -> FastAPI:
    """Construye la aplicación sobre ``engine`` (por defecto, un motor con la configuración estándar)."""
    engine = engine or InferenceEngine(EngineConfig())

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        await engine.start()
        yield
        engine.shutdown()

    app = FastAPI(
        title="Motor de inferencia H-Net",
        description="Generación de texto con H-Net (dynamic chunking sobre bytes) y baseline Qwen2.5.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.engine = engine

    @app.exception_handler(ModelNotAvailable)
    async def _model_not_available(request: Request, exc: ModelNotAvailable):
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(EngineError)
    async def _engine_error(request: Request, exc: EngineError):
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    def sse(events) -> StreamingResponse:
        async def body():
            async for event in events:
                yield to_sse(event)

        return StreamingResponse(body(), media_type="text/event-stream", headers=SSE_HEADERS)

    @app.get("/", include_in_schema=False)
    async def index():
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.get("/status")
    async def status():
        return engine.status()

    @app.get("/models", response_model=list[schemas.ModelInfoOut])
    async def models():
        return [info.__dict__ for info in engine.available_models()]

    @app.post("/models/load", responses={404: {"model": schemas.ErrorOut}})
    async def load_model(body: schemas.LoadModelBody):
        return await engine.load_model(body.name)

    @app.post(
        "/generate",
        response_model=schemas.GenerationReportOut,
        responses={400: {"model": schemas.ErrorOut}, 404: {"model": schemas.ErrorOut}},
    )
    async def generate(body: schemas.GenerateBody):
        request = GenerateRequest(body.prompt, body.to_config(), model=body.model, segment=body.segment)
        report = await engine.generate(request)
        return report.as_dict()

    @app.post("/generate/stream", responses={400: {"model": schemas.ErrorOut}})
    async def generate_stream(body: schemas.GenerateBody):
        request = GenerateRequest(body.prompt, body.to_config(), model=body.model, segment=body.segment)
        if body.model is not None:
            engine.resolve_model(body.model)  # validación previa: el error se devuelve como respuesta HTTP
        engine.validate_request(request)
        return sse(engine.stream(request))

    @app.post("/compare/stream", responses={400: {"model": schemas.ErrorOut}})
    async def compare_stream(body: schemas.CompareBody):
        for name in body.models:
            engine.resolve_model(name)
        engine.validate_request(GenerateRequest(body.prompt, body.to_config()))
        return sse(engine.compare(body.prompt, body.models, body.to_config(), segment=body.segment))

    @app.post("/segment", response_model=schemas.SegmentationOut)
    async def segment(body: schemas.SegmentBody):
        segmentation = await engine.segment(body.text, model=body.model)
        return segmentation.__dict__ | {"levels": [level.__dict__ for level in segmentation.levels]}

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Servidor del motor de inferencia H-Net.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="hnet_1stage_L", help="Modelo cargado al arrancar ('' para ninguno).")
    parser.add_argument("--warmup", type=int, default=1, help="Iteraciones de calentamiento tras cada carga.")
    parser.add_argument("--max-prompt-bytes", type=int, default=8192)
    parser.add_argument("--max-new-bytes", type=int, default=2048)
    parser.add_argument("--allow-download", action="store_true",
                        help="Permite cargar modelos ausentes de la caché local de Hugging Face.")
    args = parser.parse_args()

    import uvicorn

    config = EngineConfig(
        default_model=args.model or None,
        warmup_iterations=args.warmup,
        max_prompt_bytes=args.max_prompt_bytes,
        max_new_bytes=args.max_new_bytes,
        allow_download=args.allow_download,
    )
    uvicorn.run(create_app(InferenceEngine(config)), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
