# Entregables LaTeX — Tesis

Dos documentos en este directorio:

| Archivo | Documento | Figuras |
|---------|-----------|---------|
| `Conclusions_Avance6.tex` | Reporte técnico (paper) — "Evaluación de QLoRA Fine-Tuning vs. Modelos Clásicos para Predicción Direccional en Criptomonedas" | `figures/conclusion/` |
| `Avance7_ResumenEjecutivo.tex` | Resumen ejecutivo (Avance 7) | `figures/executive_summary/` |

## Compilar

```bash
cd public
pdflatex <archivo>.tex && pdflatex <archivo>.tex   # dos pasadas: refs + hyperlinks
```

En Windows con MiKTeX, si `pdflatex` no está en PATH usa la ruta completa
(p. ej. `%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\pdflatex.exe`).

### Prerrequisitos

TeX Live 2024+ o MiKTeX. Paquetes (todos en la instalación completa estándar):
`geometry`, `graphicx`, `booktabs`, `hyperref`, `xcolor`, `amsmath`, `amssymb`,
`caption`, `subcaption`, `float`, `tabularx`, `enumitem`, `titlesec`, `fancyhdr`,
`tcolorbox`, `babel` (spanish).

## Figuras

Las figuras están **divididas por documento**. Cada carpeta tiene un `src/` con las
fuentes editables (PNG = salida que consume LaTeX; `src/` = fuente):

```
public/
├── Conclusions_Avance6.tex
├── Avance7_ResumenEjecutivo.tex
├── README.md
└── figures/
    ├── conclusion/                 # figuras del reporte técnico
    │   ├── figN_*.png              # generadas por langgraph/optimization-results.ipynb (300 dpi)
    │   ├── fig_*.png               # diagramas (Mermaid / PlantUML)
    │   └── src/                    # fuentes: *.mmd, *.puml
    └── executive_summary/          # figuras del resumen ejecutivo (Avance 7)
        ├── fig_roadmap.png
        ├── fig_before_after.png
        ├── fig_lightsail_aws.png
        ├── fig4_accuracy_all_models.png   # compartida con conclusion/ (copia)
        ├── fig_temporal_split.png         # compartida con conclusion/ (copia)
        └── src/                    # fuentes: *.svg, icons/, build_lightsail_aws.py, README.md
```

- Las figuras `figN_*` y los diagramas de `conclusion/` las genera el notebook
  `langgraph/optimization-results.ipynb` (sus `savefig` apuntan a `conclusion/`).
- Las figuras del resumen ejecutivo son SVG: ver `figures/executive_summary/src/README.md`
  para editarlas y re-renderizarlas a PNG.
- `fig4_accuracy_all_models` y `fig_temporal_split` las usan **ambos** documentos:
  la copia canónica vive en `conclusion/` y se duplica en `executive_summary/`.
