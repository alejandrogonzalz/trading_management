# Validation Experiments — QLoRA Crypto Project

Index for the **validation/ablation experiments** that probe the QLoRA fine-tuning
results (Qwen 2.5 7B → crypto trade-direction prediction). These docs live alongside
the project steering files (`steering-*.md`) and rules (`rules/*.md`) in this
directory; they document *experiments*, not harness config.

> All experiment docs are in English (per request). The reference audit is
> `../docs/AUDIT_QLORA_88PCT.md`.

---

## Audit Context

`../docs/AUDIT_QLORA_88PCT.md` is a complete adversarial review of the 88.03% claim.
Its findings, by section:

| # | Finding | Verdict / Severity | Status |
|---|---------|--------------------|--------|
| §2 | Drawdown filter → survivorship bias (dataset of only profitable trades) | CONCERN / Medium | **Addressed by this experiment** |
| §6 | Circular TP/SL inflates profit factor (12.92) & win rate | **FAIL / HIGH** | Documented as a limitation; use ATR sim instead |
| §9 | "Billionaire test" — implied returns impossible | **FAIL / HIGH** | Same root cause as §6 |
| §4 | Inverse overfitting gap (train 70.5% < test 88%) | CONCERN / Medium | Pending (separate diagnostic) |
| §7 | ML-vs-LLM gap = information asymmetry, not a bug | CONCERN / Low | Pending (separate ablation) |
| §1, §3 | Data/label leakage, eval bugs | PASS | — |

The 88% **direction accuracy** is judged defensible *for a thesis* but conditional on
the filtered dataset; the **financial metrics are artifacts**.

---

## Experiments

### 1. No-Drawdown-Filter — **addresses Critique #2** (audit §2)
- **Status:** Prepared (code + docs on branch `experiment/no-drawdown-filter`),
  **pending execution.**
- **Goal:** Measure how much the drawdown filter inflates the 88% — i.e. the
  external validity of the headline number on unfiltered, real-world-like data.
- **Tag:** `qlora_no_drawdown_filter` · **dataset type:** `no_filter`
- **Docs:**
  - [`EXPERIMENT_NO_DRAWDOWN_FILTER.md`](./EXPERIMENT_NO_DRAWDOWN_FILTER.md) — plan,
    code changes, run commands, time/cost estimates.
  - [`SESSION_PERSISTENCE.md`](./SESSION_PERSISTENCE.md) — current state + how to resume.
  - [`RESULT_INTERPRETATION.md`](./RESULT_INTERPRETATION.md) — outcome → paper decision.

### Baseline being compared against
- **`qlora_cloud`** (filtered, 88.03%) — `../langgraph/optimization/qlora/results/qlora_cloud/result.json`
- **Zero-shot Qwen 7B** (58.49%) — `../langgraph/backtest/data/results/zero-shot-qwen7b.json`

---

> **Just want to run it?** The human-facing, start-to-finish guide is
> [`../docs/EXPERIMENT_NO_DRAWDOWN_FILTER_GUIDE.md`](../docs/EXPERIMENT_NO_DRAWDOWN_FILTER_GUIDE.md)
> — one linear, copy-paste path from a fresh GPU box to thesis-ready numbers.

## How to Use This Documentation

1. Read `EXPERIMENT_NO_DRAWDOWN_FILTER.md` for the full plan and exact commands.
2. Check `SESSION_PERSISTENCE.md` for what is done vs pending.
3. Execute the steps (generate dataset → fine-tune → compare).
4. Use `RESULT_INTERPRETATION.md` to turn the numbers into a thesis-framing decision.

## References
- Full audit: `../docs/AUDIT_QLORA_88PCT.md`
- QLoRA hyperparameter reference: `./rules/qlora-training.md`
- ML conventions (temporal split, embargo): `./rules/ml-conventions.md`
- RunPod GPU setup + cost: `../langgraph/optimization/qlora/RUNPOD_GUIDE.md`
