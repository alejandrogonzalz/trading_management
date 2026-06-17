# TODO — Incorporate New Ablation Results into the Thesis LaTeX

Tracks what's done vs pending for getting this session's experiments
(drawdown-filter ablation, symbol-anonymization ablation, feature-occlusion probe)
into the LaTeX deliverables in `public/`. Source of truth for the numbers:
`docs/qlora/AUDIT_QLORA_88PCT.md` §10, `langgraph/optimization-results.ipynb` Parts 8-9,
`.claude/SESSION_PERSISTENCE.md`.

---

## Done

- [x] **Drawdown-filter ablation** written up in `public/Addendum_Auditoria_QLoRA.tex`
      §"Ablación del filtro de drawdown" — table + figure (`fig9_drawdown_ablation.png`)
      + loss-curve figure (`fig10_loss_curves_ablation.png`, now a frozen snapshot —
      see note below) + full interpretation. Result: 84.13% vs 88.03%, -3.90pp.
- [x] **Pretraining-memorization check** written up in the same file, §"Verificación:
      'memorización' por pre-entrenamiento". Result: ruled out (test period postdates
      Qwen2.5's real pretraining cutoff).

## Pending

- [ ] **Symbol-anonymization ablation** — NOT yet in any `.tex` file. Need a new
      section in `public/Addendum_Auditoria_QLoRA.tex` (or a second addendum file)
      covering:
      - Same-hyperparams ablation, symbol→`"ASSET"` in both train export and eval.
      - Result: 88.66% (full N=8,425) vs `qlora_cloud`'s 88.03% (N=3,000) — no drop.
      - Rules out symbol-identity memorization.
      - Figure: `langgraph/optimization/qlora/results/anon_symbol_ablation.png` —
        needs copying into `public/figures/` (suggest `fig11_anon_symbol_ablation.png`,
        next free number after `fig10_loss_curves_ablation.png`) and a `\includegraphics`
        added.
      - Table: accuracy/win-rate/PF/train/val/test/gap, mirroring the drawdown-filter
        table's style (`tab:ablation-drawdown` in the existing `.tex`).

- [ ] **Feature-occlusion probe** (`heatmap`/`structure` stripped, eval-only on the
      existing `qlora_cloud` adapter, tag `qlora_strip_heatmap_structure`) —
      **RESULT IN, NOT YET EVALUATED/INTEGRATED.** Raw number: **87.60%** accuracy
      (full 8,425-sample test) vs `qlora_cloud`'s 88.03% — only **-0.43pp**, win_rate
      60.12%, PF=8.74, 0 parse errors. Small drop — leans toward "categorical fields
      are NOT the primary driver of the LLM-vs-ML gap" but this is a quick read, not
      a full analysis (eval-only ambiguity caveat below still applies — re-evaluate
      properly, don't just take the headline number at face value). Result file:
      `langgraph/optimization/qlora/results/qlora_strip_heatmap_structure/result.json`.
      **TODO: re-evaluate this result properly** (compare overfitting/gap fields, check
      per-class metrics, sanity-check against the ambiguity caveat) before integrating.
      Once properly evaluated:
      1. Read the result: `langgraph/optimization/qlora/results/qlora_strip_heatmap_structure/result.json`.
      2. Add it to `optimization-results.ipynb` (`RESULT_FILES` dict, a new Part 10
         section — table + bar chart + interpretation, same pattern as Parts 8/9).
      3. Re-execute the notebook, verify zero errors.
      4. Add a corresponding section to `public/Addendum_Auditoria_QLoRA.tex` (or a
         new addendum file) with the finding — this is the one that actually answers
         "is the 24pp LLM-vs-ML gap about categorical text encoding, or genuine
         reasoning?" (audit §7), so it's the most consequential of the three probes
         for the thesis narrative.
      5. **Important interpretive caveat to carry over**: this is an eval-only probe
         on a model trained WITH the fields present — if accuracy drops, that's
         ambiguous (could mean "model needs the fields" OR "model is confused by an
         unfamiliar prompt shape it never trained on"). State this caveat in the LaTeX
         too, don't oversell a single eval-only result as decisive. A full retrain
         with the fields stripped from training too would be the decisive version
         (not yet planned/run).

- [ ] **Equity curve redesign** (log-scale + trade-count zoom, replacing the
      non-functional `y≤20` cap) — the notebook's `fig7_equity_curves.png` changed
      significantly (log full-range + 400-trade linear zoom panels, 7 models instead
      of 5). `public/Conclusions_Avance6.tex` Figure `fig:equity` (§7 Síntesis) still
      references the OLD 5-model linear version. Decide: re-copy the updated
      `fig7_equity_curves.png` over the existing one (same filename, so the `.tex`
      reference doesn't need to change), and update the caption to mention the
      log-scale treatment and the "not real returns" caveat if desired.

- [ ] **`fig1_loss_curves.png`** also changed (now 4 panels: cloud/config3/no-filter/
      anon-symbol, was 2). `public/Conclusions_Avance6.tex` Figure `fig:loss`
      (§4 Resultados) references this file by the same name — already updated on disk
      via the notebook re-run, just confirm the caption text ("Curvas de pérdida
      QLoRA: convergencia saludable...") still reads sensibly with 4 panels instead
      of 2, or narrow the caption to refer to "cloud" specifically if the LaTeX
      section's narrative is cloud-specific.

- [ ] Decide whether `public/Addendum_Auditoria_QLoRA.tex` should be `\input{}`-ed into
      `Conclusions_Avance6.tex` directly, or stay a separate companion document — it's
      currently a standalone fragment (no `\documentclass`/`\begin{document}`), written
      that way specifically so it can be merged later. No decision made yet.

---

## Commit (prepared, not executed — user will run this)

Once the feature-occlusion probe finishes and its result/notebook/LaTeX updates above
are done, the commit covering ths work should look roughly like:

```bash
git add \
  langgraph/optimization/qlora/results/qlora_strip_heatmap_structure/ \
  langgraph/optimization-results.ipynb \
  public/Addendum_Auditoria_QLoRA.tex \
  public/figures/<any new/updated fig*.png> \
  docs/qlora/AUDIT_QLORA_88PCT.md  # if §10 gets a 3rd sub-finding appended

git commit -m "feat(qlora): add heatmap/structure feature-occlusion probe results

qlora_strip_heatmap_structure (eval-only on the existing qlora_cloud adapter,
heatmap/structure stripped from the prompt, full 8,425-sample test): <FILL IN
accuracy>% vs qlora_cloud's 88.03% (<FILL IN delta>pp). <FILL IN one-line
takeaway: rules out / supports the categorical-text-encoding hypothesis for
the 24pp LLM-vs-ML gap (audit §7) — eval-only caveat applies, see
SESSION_PERSISTENCE.md.>

Completes the third of three feature-occlusion probes investigating why
QLoRA's accuracy is so much higher than tree-based ML on the same task
(symbol anonymization and pretraining-cutoff checks were the other two,
both already ruled out as explanations)."
```

Fill in the bracketed parts from the actual result before running this.
