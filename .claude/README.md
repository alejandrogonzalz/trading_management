# Validation Experiments — QLoRA Crypto Project

Index for the **validation/ablation experiments** that probe the QLoRA fine-tuning
results (Qwen 2.5 7B → crypto trade-direction prediction). These docs live alongside
the project steering files (`steering-*.md`) and rules (`rules/*.md`) in this
directory; they document *experiments*, not harness config.

> All experiment docs are in English (per request). The reference audit is
> `../docs/qlora/AUDIT_QLORA_88PCT.md`.

---

## Audit Context

`../docs/qlora/AUDIT_QLORA_88PCT.md` is a complete adversarial review of the 88.03% claim.
Its findings, by section:

| # | Finding | Verdict / Severity | Status |
|---|---------|--------------------|--------|
| §2 | Drawdown filter → survivorship bias (dataset of only profitable trades) | CONCERN / Medium | **Addressed — see Experiment 1.** Result: -3.90pp, filter is NOT the primary driver |
| §6 | Circular TP/SL inflates profit factor (12.92) & win rate | **FAIL / HIGH** | Documented as a limitation; use ATR sim instead. The `--use-atr-tp-sl` retrain (de-circularizes training labels) is coded but **still not run** |
| §9 | "Billionaire test" — implied returns impossible | **FAIL / HIGH** | Same root cause as §6 |
| §4 | Inverse overfitting gap (train 70.5% < test 88%) | CONCERN / Medium | **Partially explained** — Experiment 1 shows the gap flips to the normal sign (+4.4pp) without the drawdown filter, suggesting the filtered model's inverted gap is a side-effect of training on cleanly-filtered data |
| §7 | ML-vs-LLM gap = information asymmetry, not a bug | CONCERN / Low | **Partially addressed** — Experiment 2 (symbol anonymization) rules out symbol-identity memorization as a factor. Experiment 3 (heatmap/structure occlusion) is in progress |
| §10 | (addendum) Pretraining memorization of test-period prices | — | **Addressed** — test period (Feb-May 2026) postdates Qwen2.5's real pretraining cutoff (~Sept 2024); ruled out |
| §1, §3 | Data/label leakage, eval bugs | PASS | — |

The 88% **direction accuracy** is judged defensible *for a thesis* but conditional on
the filtered dataset; the **financial metrics are artifacts**.

---

## Experiments

### 1. No-Drawdown-Filter — **addresses Critique #2** (audit §2)
- **Status:** ✅ **Completed (2026-06-17).**
- **Goal:** Measure how much the drawdown filter inflates the 88% — i.e. the
  external validity of the headline number on unfiltered, real-world-like data.
- **Result:** **84.13%** (vs 88.03%, -3.90pp) — between Scenario B and C (see
  `RESULT_INTERPRETATION.md`). The auditor's predicted 60-70% (§5) did not materialize.
  Most informative finding: the overfitting gap flips sign (-17.5pp → +4.4pp) without
  the filter — evidence it's a quality control, not an inflation artifact.
- **Tag (actual):** `qlora_no_drawdown` (planning docs say `qlora_no_drawdown_filter` —
  drifted during execution) · **dataset type:** `no_filter`
- **Docs:**
  - [`EXPERIMENT_NO_DRAWDOWN_FILTER.md`](./EXPERIMENT_NO_DRAWDOWN_FILTER.md) — plan,
    code changes, run commands, time/cost estimates.
  - [`SESSION_PERSISTENCE.md`](./SESSION_PERSISTENCE.md) — current state + results.
  - [`RESULT_INTERPRETATION.md`](./RESULT_INTERPRETATION.md) — outcome → paper decision (applied).
  - `../langgraph/optimization-results.ipynb` Part 8 — charts + write-up.

### 2. Symbol Anonymization — **addresses part of Critique #7** (audit §7, information asymmetry)
- **Status:** ✅ **Completed (2026-06-17).**
- **Goal:** Test whether the model relies on symbol-specific pretrained associations
  (e.g. "knowing" BTC) rather than reasoning from the visible indicators.
- **Result:** **88.66%** (full 8,425-sample test) vs `qlora_cloud`'s 88.03% (3,000-sample
  subset) — no drop. Rules out symbol-identity memorization.
- **Tag:** `qlora_anon_symbol` · same hyperparams + dataset as `qlora_cloud`, symbol name
  replaced with `"ASSET"` in both training export and eval (`--anonymize-symbol` flag).
- **Docs:** `SESSION_PERSISTENCE.md` "Follow-up Experiment" section;
  `../langgraph/optimization-results.ipynb` Part 9.

### 3. Feature Occlusion (heatmap/structure) — **addresses the rest of Critique #7**
- **Status:** 🔄 In progress (eval-only probe, no retrain, on the existing `qlora_cloud`
  adapter — `--strip-fields heatmap,structure`).
- **Goal:** Isolate how much of the 24pp LLM-vs-tree-model gap is the categorical
  text encoding of `heatmap`/`structure` (which trees only see as lossy ordinal ints)
  vs genuine reasoning capability.
- **Tag:** `qlora_strip_heatmap_structure`. Not yet in the notebook.

### Baseline being compared against
- **`qlora_cloud`** (filtered, 88.03%) — `../langgraph/optimization/qlora/results/qlora_cloud/result.json`
- **Zero-shot Qwen 7B** (58.49% filtered, 56.49% no-filter) — `../langgraph/backtest/data/results/zero-shot-qwen7b*.json`

---

> **Just want to run it?** The human-facing, start-to-finish guide is
> [`../docs/qlora/EXPERIMENT_NO_DRAWDOWN_FILTER_GUIDE.md`](../docs/qlora/EXPERIMENT_NO_DRAWDOWN_FILTER_GUIDE.md)
> — one linear, copy-paste path from a fresh GPU box to thesis-ready numbers.

## How to Use This Documentation

1. Read `EXPERIMENT_NO_DRAWDOWN_FILTER.md` for the full plan and exact commands.
2. Check `SESSION_PERSISTENCE.md` for what is done vs pending.
3. Execute the steps (generate dataset → fine-tune → compare).
4. Use `RESULT_INTERPRETATION.md` to turn the numbers into a thesis-framing decision.

## References
- Full audit: `../docs/qlora/AUDIT_QLORA_88PCT.md`
- LaTeX write-up checklist (what's in `public/*.tex` vs still pending): `./TODO_LATEX_UPDATES.md`
- QLoRA hyperparameter reference: `./rules/qlora-training.md`
- ML conventions (temporal split, embargo): `./rules/ml-conventions.md`
- RunPod GPU setup + cost: `../langgraph/optimization/qlora/RUNPOD_GUIDE.md`
