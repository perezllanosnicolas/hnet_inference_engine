# hnet_inference_engine (v1)

> **Versión v1 (primera versión funcional).** Prototipo de referencia para validar la arquitectura del motor y obtener las primeras medidas. Funciona con batch = 1 y no incluye todavía batching, optimización de la decodificación ni imagen Docker. La guía completa (contexto del TFG, estructura del código, ejecución en otro equipo, resultados y trabajo futuro) se encuentra en [INICIO.md](INICIO.md).

Motor de inferencia en GPU para **H-Net**, un modelo de lenguaje sin tokenizador que opera directamente sobre bytes y agrupa la secuencia en *chunks* aprendidos mediante *dynamic chunking*. El motor ofrece una interfaz común para H-Net y para un baseline Transformer denso (Qwen2.5) y los evalúa con el mismo código de generación, muestreo y medición.

Trabajo Fin de Grado, Grado en Ingeniería de Computadores (URJC).

## Estado (v1)

- Carga de los checkpoints oficiales de H-Net (`cartesia-ai/hnet_1stage_L`, `hnet_2stage_L`, ...) y de Qwen2.5-0.5B / 1.5B.
- Generación incremental (prefill y decodificación byte a byte) con muestreo configurable (temperatura, top-k, top-p, penalización por repetición) y condiciones de parada.
- Métricas por petición: tiempo hasta la primera unidad (TTFT), velocidad de decodificación en bytes/s y en unidades nativas/s, bytes por unidad, pico de VRAM y tasa de compresión por etapa de H-Net.
- Servicio de inferencia (`src/core/engine.py`): un modelo residente en GPU, cola FIFO de peticiones con batch = 1 y streaming de texto.
- API REST con FastAPI (`src/api/`) y página web de demostración: generación en streaming, comparación de modelos con el mismo prompt y visualización de la segmentación (chunks de H-Net frente a tokens BPE).
- Benchmarks reproducibles de throughput y de pico de memoria (`scripts/`).

Pendiente para versiones posteriores: batching de varias secuencias, optimización de la decodificación de H-Net (actualmente limitada por la CPU), caché propia e imagen Docker. La lista completa figura en la última sección de [INICIO.md](INICIO.md).

## Requisitos

- GPU NVIDIA con arquitectura Ampere o posterior (bf16), CUDA Toolkit 12.x.
- Linux o WSL 2 (probado en Ubuntu 24.04), Python 3.12.

Las versiones de las dependencias están fijadas. La instalación debe realizarse sin resolución de dependencias, ya que pip tiende a sustituir PyTorch por compilaciones incompatibles:

```bash
python3.12 -m venv .venv
.venv/bin/pip install --no-deps --no-build-isolation -r requirements.txt
```

## Ejecución de la aplicación

```bash
.venv/bin/python -m src.api.app --model hnet_1stage_L --port 8000
```

Tras la carga y el calentamiento del modelo (unos 20 s la primera vez), la página de demostración queda disponible en <http://127.0.0.1:8000> y la documentación interactiva de la API, en <http://127.0.0.1:8000/docs>.

Opciones principales:

| Opción | Descripción |
|---|---|
| `--model` | Modelo cargado al arrancar (`hnet_1stage_L`, `hnet_2stage_L`, `Qwen2.5-0.5B`, `Qwen2.5-1.5B`). |
| `--allow-download` | Permite descargar desde Hugging Face los modelos ausentes de la caché local. |
| `--warmup` | Iteraciones de calentamiento tras cada carga (0 para desactivarlo). |
| `--max-prompt-bytes`, `--max-new-bytes` | Límites de longitud de las peticiones. |

### Página de demostración

- **Generación**: texto generado en streaming con sus métricas y, opcionalmente, la segmentación del resultado.
- **Comparación**: ejecuta el mismo prompt, con los mismos parámetros y semilla, sobre varios modelos en secuencia y resume sus métricas en una tabla.
- **Segmentación**: muestra cómo divide cada modelo un texto: chunks de cada etapa en H-Net y tokens BPE en Qwen.
- **Cómo funciona**: diagramas del flujo de H-Net (encoder, routing, red principal, dechunk, decoder) y de la arquitectura del servicio, y guía de lectura de las métricas.

Resultados visuales: embudo de compresión por etapa (bytes frente a chunks), gráficas de barras comparativas (velocidad, TTFT, VRAM, bytes por unidad) y ejemplos de prompt (inglés, español, código, Unicode, números). Cada pestaña admite enlace directo (`/#generate`, `/#compare`, `/#segment`, `/#about`).

### API REST

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/status` | Modelo cargado, peticiones en cola y memoria de GPU. |
| GET | `/models` | Catálogo de modelos y disponibilidad local. |
| POST | `/models/load` | Carga un modelo (sustituye al residente). |
| POST | `/generate` | Generación completa con métricas (JSON). |
| POST | `/generate/stream` | Generación en streaming (Server-Sent Events). |
| POST | `/compare/stream` | Comparación de varios modelos (Server-Sent Events). |
| POST | `/segment` | Segmentación de un texto. |

Ejemplo:

```bash
curl -s http://127.0.0.1:8000/generate -H 'Content-Type: application/json' \
  -d '{"prompt": "The capital of France is", "max_new_bytes": 64, "temperature": 0}'
```

## Benchmarks y pruebas

```bash
.venv/bin/python scripts/eval_throughput.py --models hnet_1stage_L Qwen2.5-0.5B --prompt-bytes 256 1024
.venv/bin/python scripts/eval_memory_peak.py --models hnet_1stage_L Qwen2.5-0.5B
.venv/bin/python -m unittest tests.test_engine tests.test_api -v
```

Los resultados de los benchmarks se guardan en `results/` en formato JSON junto con la descripción del entorno (GPU, versiones y commit). Las figuras de esta documentación se regeneran con `scripts/generar_figuras.py`.

### Resultados de la v1 (RTX 4060 Laptop, 8 GB, bf16, batch = 1, voraz)

Velocidad de decodificación (mediana de 5 repeticiones, 256 bytes generados):

![Velocidad de decodificación](docs/figuras/decodificacion.svg)

Pico de VRAM durante la generación:

![Pico de VRAM](docs/figuras/memoria_pico.svg)

El análisis completo (TTFT, compresión de H-Net, descripción de la batería y limitaciones) figura en [INICIO.md](INICIO.md).

## Estructura

```
src/core/engine.py              servicio de inferencia (modelos, cola, streaming, comparación)
src/api/                        API REST (FastAPI) y página de demostración
src/models/wrappers/            interfaz común (base.py) y envoltorios de H-Net y Qwen
src/models/architectures/hnet/  código de H-Net (goombalab/hnet) con correcciones mínimas
src/models/architectures/hnet_compat.py  adaptadores de compatibilidad de versiones
src/utils/metrics.py            utilidades de medición reproducible
scripts/                        benchmarks, generación de figuras y prueba de humo de la arquitectura
docs/figuras/                   figuras SVG de los resultados de la v1
configs/                        configuraciones de H-Net y texto de referencia de los benchmarks
tests/                          pruebas unitarias y de coherencia
```

## Referencias

- Hwang, Wang y Gu, *Dynamic Chunking for End-to-End Hierarchical Sequence Modeling* (2025). Código: <https://github.com/goombalab/hnet>.
- Checkpoints: <https://huggingface.co/collections/cartesia-ai/h-net>.
