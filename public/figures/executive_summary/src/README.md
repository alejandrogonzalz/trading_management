# Fuentes de figuras — Resumen Ejecutivo (Avance 7)

Convención: **las fuentes editables viven aquí (`src/`)** y los **PNG renderizados
viven un nivel arriba (`executive_summary/`)**, que es lo que consume el `.tex` con
`\includegraphics{figures/executive_summary/<nombre>.png}`.

Ningún `.tex` referencia archivos de `src/` directamente. Si editas una fuente,
regenera el PNG (a mano) y se actualiza en el documento.

## Figuras

| Fuente | Salida (`../`) | Usada en | Notas |
|--------|----------------|----------|-------|
| `fig_roadmap.svg` | `fig_roadmap.png` | §4 | SVG a mano (viewBox 1100×470) |
| `fig_before_after.svg` | `fig_before_after.png` | §3 | SVG a mano (viewBox 1080×470) |
| `fig_lightsail_aws.svg` | `fig_lightsail_aws.png` | §4 | **Generado** por `build_lightsail_aws.py` (incrusta `icons/`); viewBox 1060×720 |
| `icons/aws_lightsail.svg` | — | — | Icono oficial AWS Lightsail (insumo) |
| `icons/aws_s3.svg` | — | — | Icono oficial Amazon S3 (insumo) |
| `icons/binance_mark.png` | — | — | Marca de Binance recortada (insumo) |

> `fig4_accuracy_all_models.png` y `fig_temporal_split.png` (también en esta carpeta)
> son **compartidas con el reporte técnico**: su fuente vive en `../../conclusion/src/`
> (la Mermaid) o la genera el notebook. Aquí solo está la copia PNG.

## Regenerar

```bash
cd public/figures/executive_summary/src

# 1) Solo si cambian iconos o el layout del diagrama AWS:
python build_lightsail_aws.py        # reescribe fig_lightsail_aws.svg

# 2) Renderizar SVG -> PNG a mano (Inkscape, navegador headless, etc.),
#    guardando el PNG en la carpeta padre con el mismo nombre y respetando el
#    viewBox para no dejar margen blanco. Ejemplo con Edge/Chrome headless:
#    msedge --headless --screenshot=../fig_roadmap.png \
#           --window-size=1100,470 --default-background-color=FFFFFFFF \
#           "file:///<ruta-abs>/fig_roadmap.svg"
```

Tamaños (`--window-size` = viewBox del SVG, para PNG sin margen):
`fig_roadmap` 1100×470 · `fig_before_after` 1080×470 · `fig_lightsail_aws` 1060×720.
