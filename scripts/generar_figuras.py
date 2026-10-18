"""
Genera las figuras SVG de los resultados de la v1 a partir de los JSON de ``results/``.

No requiere dependencias adicionales: los gráficos de barras se escriben
directamente como SVG, de modo que puedan incluirse en el Markdown del
repositorio.

Ejemplo:
    .venv/bin/python scripts/generar_figuras.py \\
        --throughput results/throughput_20261018-134404.json \\
        --memoria results/memory_peak_20261018-141542.json
"""

import argparse
import json
import os
from xml.sax.saxutils import escape

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
COLORES = ["#2a6fb0", "#e08a1e", "#3a9d5d", "#b0413e", "#7b5ea7"]
NOMBRES = {
    "hnet_1stage_L": "H-Net 1 etapa (L)",
    "hnet_2stage_L": "H-Net 2 etapas (L)",
    "Qwen2.5-0.5B": "Qwen2.5-0.5B",
    "Qwen2.5-1.5B": "Qwen2.5-1.5B",
}


def barras_agrupadas(titulo, etiqueta_y, grupos, series, valores, ruta, formato="{:.0f}"):
    """Escribe un gráfico de barras agrupadas: ``valores[serie][grupo]``."""
    ancho, alto = 760, 400
    izq, der, arriba, abajo = 70, 20, 60, 80
    w, h = ancho - izq - der, alto - arriba - abajo
    maximo = max(v for s in valores for v in s) * 1.12
    paso = 10 ** (len(str(int(maximo))) - 1)
    for m in (1, 2, 2.5, 5, 10):
        if maximo / (paso * m / 10) <= 6:
            paso = paso * m / 10
            break
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ancho} {alto}" '
         f'font-family="sans-serif" font-size="12">',
         f'<rect width="{ancho}" height="{alto}" fill="#ffffff"/>',
         f'<text x="{ancho / 2}" y="26" text-anchor="middle" font-size="15" '
         f'font-weight="bold" fill="#222">{escape(titulo)}</text>']
    y = 0.0
    while y <= maximo:
        py = arriba + h - y / maximo * h
        o.append(f'<line x1="{izq}" y1="{py:.1f}" x2="{izq + w}" y2="{py:.1f}" stroke="#ddd"/>')
        o.append(f'<text x="{izq - 6}" y="{py + 4:.1f}" text-anchor="end" fill="#444">{y:g}</text>')
        y += paso
    o.append(f'<text x="16" y="{arriba + h / 2}" transform="rotate(-90 16 {arriba + h / 2})" '
             f'text-anchor="middle" fill="#444">{escape(etiqueta_y)}</text>')
    ancho_g = w / len(grupos)
    ancho_b = ancho_g * 0.8 / len(series)
    for i, g in enumerate(grupos):
        x0 = izq + i * ancho_g + ancho_g * 0.1
        for j, _ in enumerate(series):
            v = valores[j][i]
            bh = v / maximo * h
            x = x0 + j * ancho_b
            o.append(f'<rect x="{x:.1f}" y="{arriba + h - bh:.1f}" width="{ancho_b - 2:.1f}" '
                     f'height="{bh:.1f}" fill="{COLORES[j % len(COLORES)]}"/>')
            o.append(f'<text x="{x + (ancho_b - 2) / 2:.1f}" y="{arriba + h - bh - 4:.1f}" '
                     f'text-anchor="middle" font-size="10" fill="#222">{formato.format(v)}</text>')
        o.append(f'<text x="{izq + (i + 0.5) * ancho_g:.1f}" y="{arriba + h + 18}" '
                 f'text-anchor="middle" fill="#222">{escape(g)}</text>')
    o.append(f'<line x1="{izq}" y1="{arriba + h}" x2="{izq + w}" y2="{arriba + h}" stroke="#444"/>')
    lx = izq
    for j, s in enumerate(series):
        o.append(f'<rect x="{lx}" y="{alto - 28}" width="12" height="12" fill="{COLORES[j % len(COLORES)]}"/>')
        o.append(f'<text x="{lx + 17}" y="{alto - 18}" fill="#222">{escape(s)}</text>')
        lx += 22 + 7 * len(s)
    o.append("</svg>")
    with open(ruta, "w", encoding="utf-8") as f:
        f.write("\n".join(o))
    print(f"Figura escrita: {os.path.relpath(ruta, ROOT)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--throughput", required=True)
    ap.add_argument("--memoria", required=True)
    ap.add_argument("--salida", default=os.path.join(ROOT, "docs", "figuras"))
    a = ap.parse_args()
    os.makedirs(a.salida, exist_ok=True)

    with open(a.throughput, encoding="utf-8") as f:
        thr = json.load(f)["results"]
    with open(a.memoria, encoding="utf-8") as f:
        mem = json.load(f)["results"]
    modelos = list(dict.fromkeys(r["model"] for r in thr))
    longitudes = sorted({r["prompt_bytes"] for r in thr})
    grupos = [f"prompt {n} B" for n in longitudes]
    tabla = {(r["model"], r["prompt_bytes"]): r for r in thr}

    def serie(clave):
        return [[tabla[(m, n)]["summary"][clave]["median"] for n in longitudes] for m in modelos]

    nombres = [NOMBRES.get(m, m) for m in modelos]
    barras_agrupadas("Velocidad de decodificación (mediana, batch = 1)", "bytes por segundo",
                     grupos, nombres, serie("decode_bytes_per_s"),
                     os.path.join(a.salida, "decodificacion.svg"), "{:.1f}")
    ttft = [[v * 1000 for v in s] for s in serie("prefill_time_s")]
    barras_agrupadas("Tiempo hasta la primera unidad (TTFT, mediana)", "milisegundos",
                     grupos, nombres, ttft, os.path.join(a.salida, "ttft.svg"), "{:.0f}")

    pesos = {r["model"]: r["weights_mib"] for r in mem}
    picos = {(r["model"], c["prompt_bytes"]): c["generate"]["peak_allocated_mib"]
             for r in mem for c in r["configs"]}
    lmem = sorted({n for (_, n) in picos})
    barras_agrupadas("Pico de VRAM durante la generación (256 B generados)", "MiB (max_memory_allocated)",
                     [f"prompt {n} B" for n in lmem], [NOMBRES.get(m, m) for m in pesos],
                     [[picos[(m, n)] for n in lmem] for m in pesos],
                     os.path.join(a.salida, "memoria_pico.svg"))

    comp = {}
    for r in thr:
        if r["model"].startswith("hnet"):
            c = r["extra"]["prefill_compression_per_stage"]
            comp[(r["model"], r["prompt_bytes"])] = c
    etiquetas = ["1 etapa", "2 etapas, etapa 0", "2 etapas, etapa 1"]
    series = [f"prompt {n} B" for n in longitudes]
    valores = []
    for n in longitudes:
        c1, c2 = comp[("hnet_1stage_L", n)], comp[("hnet_2stage_L", n)]
        valores.append([c1[0], c2[0], c2[1]])
    barras_agrupadas("Compresión de H-Net en el prefill (bytes por chunk)", "bytes por chunk",
                     etiquetas, series, valores, os.path.join(a.salida, "compresion.svg"), "{:.2f}")


if __name__ == "__main__":
    main()
