# Thesis Document — Writing & Style Guide

Applies to the LaTeX deliverable `public/Conclusions_Avance6.tex` and any thesis
prose. Distilled from the 83 reviewer (jordigonzalez) annotations + decisions made
during the rewrite. **Follow these whenever editing the thesis.**

The full annotation list lives in `tmp/insights/reviewer_comments_avance6.md` (outside
the repo). This file is the actionable summary.

---

## Mandatory style rules (reviewer-enforced)

1. **"Tabla", never "Cuadro".** babel-spanish auto-names table floats "Cuadro"; the
   preamble forces "Tabla" via `\renewcommand{\tablename}{Tabla}` +
   `\addto\captionsspanish`. Do not remove that.
2. **Exactitud ≠ Precisión.** *Exactitud* (accuracy) = global hit rate, one number.
   *Precisión* (precision) = TP/(TP+FP), one number per class. The headline 84/88 % is
   **exactitud direccional**. Reserve "precisión" for per-class metrics only. Never
   synonyms (reviewer flagged 3×).
3. **No code names / paths / ports in the body.** No function names, `*.py`, `*.yaml`,
   `/api/*`, `:8001`, run tags (`qlora_no_drawdown`, `cloud`), `Nginx`, `FastAPI`,
   `Unsloth`, `--flags`, `cxx11abi`. Describe conceptually. If a class is renamed, the
   paper must not need editing.
4. **No dollar amounts in the body.** No `$1.39/hr`, `$32`, `$5B cap`. Aggregate cost
   facts (e.g. "<8 USD per training run") may appear once, in the infra appendix only.
5. **Prose, not loose bullet lists.** Explain findings in connected paragraphs. A list
   is acceptable only when items are genuinely parallel; otherwise convert to prose
   (reviewer flagged future-work and limitations lists specifically).
6. **Foreign terms in italics, defined on first use.** *fine-tuning*, *zero-shot*,
   *hindsight*, *timeframe*, *profit factor*, *drawdown*, *slippage*, *walk-forward*.
7. **Describe every figure/table in the body.** Each `\includegraphics`/table needs
   prose stating *what is concluded* from it — never drop a figure unexplained.
8. **Diagrams use a formal framework (UML/Mermaid).** Conceptual diagrams are authored
   as Mermaid (`public/figures/src/*.mmd`) or PlantUML (`*.puml`) and rendered to PNG
   keeping the **same filename** the `.tex` references.
9. **One single Conclusión** (singular), aligned to the objective. Structure:
   Introducción → Desarrollo → Resultados → Discusión → Conclusión → Trabajo futuro →
   Anexos. "Investigación a largo plazo" / infra recommendations go AFTER the
   conclusion (appendix), not in the body.
10. **Define sets formally** (vectors in $\mathbb{R}^n$, etc.) where relevant.
11. **IEEE citations with `\cite{}`.** If something isn't cited, it isn't in the
    bibliography. Prefer 2024–2026 state-of-the-art sources.

---

## Canonical numbers (verify against result.json, never from memory)

Source of truth: `langgraph/optimization/qlora/results/<tag>/result.json`. Always read
`metrics_atr` (legitimate) for financial metrics, NOT `metrics` (circular hindsight).

| Model (display name) | Tag | Dataset | Acc. | PF (ATR) |
|----------------------|-----|---------|------|----------|
| QLoRA **base** (filtrado) | `qlora_config3` | filtered 56k | 87.87 % | n/d (hindsight only) |
| QLoRA **anonimizado** (filtrado) | `qlora_anon_symbol` | filtered 56k | 88.66 % | 4.63 |
| QLoRA **final** (todas las muestras) | `qlora_no_drawdown` | all 75k | 84.13 % | 2.09 |
| Zero-shot Qwen 7B | — | all 75k | 56.49 % | 0.70 |
| XGBoost / Random Forest | — | — | 62.7 / 61.9 % | 2.34 / 2.23 |
| LSTM | — | — | 51.51 % | 1.49 |

- **McNemar (final vs zero-shot, full 11,294 paired):** $b=4060$, $c=938$, $\chi^2=1949$, $p\approx0$.
- **Per-symbol (final):** 81.8 % (DOT) – 86.4 % (SOL), total 84.1 % over 11,294.
- Naming scheme is **base / anonimizado / final** — do NOT reintroduce "cloud",
  "config3", "no drawdown filter", "sin filtro" as a model name (it's fine as the
  technical description "sin el filtro de drawdown").

---

## Conceptual conventions (define once, cross-reference after)

- **Circularidad**: defined in §Métricas (`\label{sec:metrics}`). TP/SL derived from
  future prices → model replicates → simulator "confirms" → inflated PF. Mitigated by
  ATR forward-looking simulation. Don't re-explain the mechanism elsewhere; reference it.
- **Filtro de drawdown**: defined in §Pipeline/Etiquetado (one of 5 quality filters).
  Discards samples whose SL would trigger before TP → survivorship bias. Quantified
  only in the ablation §`sec:ablation-drawdown`.
- **ATR exits**: `TP = entry ± 1.5×ATR`, `SL = entry ∓ 1.0×ATR` → 1.5:1 R:R, breakeven
  at 40 % win rate, same ATR multiple the labeler uses (coherence).
- **Which PF matters**: the **final** model's 2.09 (all samples, closest to production),
  NOT the anonimizado's 4.63 (inflated by the easier filtered subset).
- **Purpose of fine-tuning**: the model is the reasoning core of a **ReAct agent +
  LangGraph** production app → structured JSON output and reasoning coherence are
  deliberate design properties, not just direction accuracy.
- **QLoRA has no ROC/PR/reliability**: it emits only 2 confidence values → no continuous
  probability. Those curves apply to RF/XGBoost only; QLoRA's equivalent is the
  confusion matrix. (See `docs/FAQ.md`.)

---

## LaTeX gotchas (learned the hard way)

- **`\,\%` inside math mode crashes the build** ("Incompatible glue units" via
  babel-spanish `\es@sppercent`). Keep `%` out of `$...$`: write `0.9995` then `(99.95\,\%)`
  in text. The preamble has `\AtBeginDocument{\def\%{\char37\relax}}` as a guard.
- **`figure*` (2-col floats) cause big whitespace gaps** — they can only float to a page
  top. Use `figure*` only for genuinely wide figures (accuracy bar, McNemar, trade
  metrics, equity, 2-panel loss). Single-panel charts should be 1-column `figure`.
- **Compile with `-halt-on-error`** to catch real errors; the "Incompatible glue units"
  tabularx warning is non-fatal if a PDF is produced, but the math-mode `\%` one IS fatal.
- Build: `cd public && pdflatex -interaction=nonstopmode Conclusions_Avance6.tex` twice
  (cross-refs). Artifacts (`*.aux/log/out/pdf`, `*.bak`) are gitignored in `public/`.

---

## Figure provenance

- **Data charts** (`fig2`–`fig5`, `fig8`–`fig13`): generated by
  `langgraph/optimization-results.ipynb`, saved to `public/figures/`. Re-run the
  notebook to regenerate. They read `metrics_atr`, drop the legacy `cloud` run and the
  duplicate LSTM, and carry no horizontal gridlines.
- **Diagrams** (`fig_pipeline_etl`, `fig_temporal_split`, `fig_lightsail`,
  `fig_synthesis`): authored in `public/figures/src/` (`.mmd` Mermaid / `.puml`
  PlantUML), rendered with `mermaid-cli` / `plantuml`, same output filename.
