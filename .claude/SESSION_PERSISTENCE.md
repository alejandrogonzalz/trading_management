# Session State — No-Drawdown-Filter Experiment

**Preparation date:** 2026-06-16
**Execution date:** 2026-06-17
**Status:** ✅ **COMPLETED.** Trained, evaluated, paired zero-shot re-run done, results in
the notebook + audit doc. See also the symbol-anonymization follow-up experiment
(also completed) below.

> Naming note: the executed model tag is **`qlora_no_drawdown`** (not
> `qlora_no_drawdown_filter` as originally planned in this doc / the EXPERIMENT doc).
> All paths below use the actual tag. Update any copy-pasted commands accordingly.

---

## Code Status

| Component | Status | File / Notes |
|-----------|--------|--------------|
| Labeler — conditional drawdown filter | ✅ | `langgraph/backtest/ingestion/labeler.py` — `apply_drawdown_filter=True` param on `label_candle` + `generate_labeled_dataset` |
| Data pipeline — flag plumbed | ✅ | `langgraph/backtest/pipeline.py` — `DataPipeline(apply_drawdown_filter=...)` |
| CLI — `--no-drawdown-filter` / `--output` | ✅ | `langgraph/cli.py` — `prepare-dataset` |
| Training — `--dataset-type` / `--dataset` | ✅ | `langgraph/optimization/qlora/train_qlora.py` — defaults reproduce original run |
| `run_cloud.sh` | ✅ no change | already forwards `--dataset-type no_filter` to the trainer |
| Fresh-instance bootstrap | ✅ rewritten | `optimization/qlora/user_data.sh` — steps 7-9 were broken (referenced `setup_unsloth_pod.sh` and `run_experiment_no_drawdown_filter.sh`, neither of which exist on this branch); rewritten to: verify dvc/unsloth/CUDA, `dvc pull` the exact `.dvc` pointers, install TA-Lib (missing from the training image), smoke test, then launch `run_cloud.sh` |
| Unit tests | ✅ passing | `tests/backtest/test_backtest.py` + `tests/optimization/test_qlora.py`. 79/81 pass (2 pre-existing failures: `xgboost` not installed in the GPU training image — unrelated to this experiment) |
| Ruff lint + format | ✅ clean | all changed files pass `ruff check` + `ruff format --check` |
| Documentation (`.claude/`) | ✅ | this file + `README.md`, `RESULT_INTERPRETATION.md`, `EXPERIMENT_NO_DRAWDOWN_FILTER.md` — all updated post-execution |

---

## Measured Results (2026-06-17)

| Metric | `qlora_cloud` (filtered) | `qlora_no_drawdown` (no-filter) | Δ |
|--------|---------------------------|----------------------------------|---|
| Direction accuracy | 88.03% | **84.13%** | **-3.90pp** |
| Win rate | 61.67% | 50.55% | -11.12pp |
| Profit factor (circular) | 12.92 | 4.13 | — |
| Max drawdown | 12.3% | 39.7% | — |
| Sharpe | 15.79 | 9.40 | — |
| Train/Val/Test acc | 70.5% / 86.5% / 88.0% | 88.5% / 87.5% / 84.1% | — |
| Overfitting gap | **-17.5pp** (healthy) | **+4.4pp** (normal direction) | sign flip |
| N evaluated | 3,000 (strided `--max-eval`) | 11,294 (full test, no cap) | not identical N |

**Decision (per `RESULT_INTERPRETATION.md`'s table)**: -3.90pp falls just under the
">85% = robust" cutoff (Scenario C) and above the "70-75% = major confounder" cutoff
(Scenario A) — closer to **Scenario C's conclusion** than B's: the drawdown filter is
**not** the primary driver of the 88.03% headline, and the auditor's 60-70% prediction
(audit §5) did not materialize. The gap-sign flip (healthy negative → normal positive)
is itself informative: removing the filter introduces real (if mild) overfitting,
consistent with the filter being a legitimate quality signal rather than an artifact
that was inflating the number.

**Paired zero-shot re-run** (`zero-shot-qwen7b-no-drawdown`, Qwen2.5-7B-Instruct-Turbo
via Together AI, full 11,294-sample test, concurrency=8): **56.49%** vs 58.49% filtered
(-2.00pp). Fine-tuning's advantage over zero-shot **holds up** on the no-filter data:
+29.5pp (filtered) vs +27.6pp (no-filter) — see `optimization-results.ipynb` Part 8.

All of this is written up in `optimization-results.ipynb` Part 8 and
`docs/qlora/AUDIT_QLORA_88PCT.md` §10.

## Pending Tasks

- [x] `dvc pull backtest/data/candles`
- [x] Generate unfiltered dataset → **75,289 samples (+34.1% vs 56,161)**
- [x] `dvc add` + push the unfiltered dataset
- [x] Fine-tune `qlora_no_drawdown` (RunPod A100)
- [x] Re-run zero-shot on the unfiltered test split (paired) — **56.49%**
- [x] DVC-track + push the new model and dataset
- [x] Apply the `RESULT_INTERPRETATION.md` decision table → updated
- [ ] Paired McNemar/t-test (`compare-stats`) between `qlora_no_drawdown` and
      `zero-shot-qwen7b-no-drawdown` — numbers exist, statistical test not yet run
- [ ] `qlora_no_drawdown` evaluated on the *filtered* test set (Scenario D check, see
      `RESULT_INTERPRETATION.md` "Two Evaluation Surfaces") — not done (moot here since
      we landed close to Scenario C, not D)

---

## Follow-up Experiment: Symbol Anonymization (also completed, 2026-06-17)

Separate from the drawdown-filter ablation, but came out of the same investigation
("why is 88% so high, are we missing something" — see `docs/qlora/AUDIT_QLORA_88PCT.md`
§10). Tests whether the model relies on symbol-specific pretrained associations.

- **Tag:** `qlora_anon_symbol`. Same hyperparams as `qlora_cloud`, same filtered dataset.
  Symbol name replaced with `"ASSET"` in **both** training-data export and eval (new
  `--anonymize-symbol` flag on `train_qlora.py`, plumbed through `backtest/export.py`'s
  `build_training_example()`/`export_training_data()` too — see `_occlude()` in
  `train_qlora.py`).
- **Result:** **88.66%** (full 8,425-sample test) vs `qlora_cloud`'s 88.03% (3,000-sample
  subset) — no drop, if anything slightly higher on a more robust N. Train/val/test =
  87.0%/92.0%/88.66%, gap -1.66pp (healthy).
- **Conclusion:** rules out symbol-identity memorization as an explanation for the high
  accuracy — the model performs identically without knowing which real asset it's
  looking at. Written up in `optimization-results.ipynb` Part 9.
- A third occlusion probe (`--strip-fields heatmap,structure`, eval-only on the existing
  `qlora_cloud` adapter — no retrain) is in progress as of this writing; not yet in the
  notebook.

---

## Decisions Made (and why)

1. **Hyperparameters = `qlora_cloud` baseline** (`lr=2e-5, rank=16, alpha=32, epochs=2`),
   NOT the prompt's `config3` `lr=1e-5`. Reason: a single-variable ablation must hold
   everything but the dataset constant, and the baseline we compare against (88.03%)
   is `qlora_cloud`. Either config is defensible *if held constant on both sides*;
   matching the baseline is the cleaner comparison. (Documented in EXPERIMENT doc §4.)
2. **Only the drawdown filter is removed.** ADX≥15, volume≥0.5, R:R≥1.0, directional
   clarity, and the whipsaw filter are unchanged. Same lookahead (24), same temporal
   split + embargo.
3. **Separate output files** so nothing overwrites the production pipeline:
   - dataset → `dataset_no_drawdown_filter.jsonl` (not `dataset.jsonl`)
   - chat export → `training_data_no_filter/` (not `training_data/`)
   - model → `backtest/data/models/qlora_no_drawdown/`
4. **Backward compatibility:** all new flags default to the original behavior; running
   the existing commands without new flags is byte-for-byte unchanged.
5. **No `--max-eval` cap on either the no-filter or anon-symbol run** — both evaluate the
   full test set, unlike `qlora_cloud`'s original strided 3,000. This means N differs
   across comparisons (noted explicitly wherever it matters); the larger N is the more
   statistically robust side of each comparison, not a weaker one.

---

## Quick Commands (as actually run)

```bash
cd langgraph

# Generate unfiltered dataset (CPU, ~15s on already-pulled candles)
python -m cli prepare-dataset --no-drawdown-filter
# → backtest/data/labeled/dataset_no_drawdown_filter.jsonl (75,289 samples)

# Train (RunPod A100, hyperparams match qlora_cloud baseline)
nohup bash optimization/qlora/run_cloud.sh \
  --tag qlora_no_drawdown --dataset-type no_filter \
  > optimization/qlora/logs/qlora_no_drawdown.log 2>&1 & echo "PID: $!"

# Symbol-anonymization follow-up (filtered dataset, same hyperparams)
nohup bash optimization/qlora/run_cloud.sh \
  --tag qlora_anon_symbol --anonymize-symbol \
  > optimization/qlora/logs/qlora_anon_symbol.log 2>&1 & echo "PID: $!"

# Feature-occlusion probe (eval-only, no retrain, ~1h on the full test set)
python optimization/qlora/train_qlora.py \
  --eval-only backtest/data/models/qlora_cloud \
  --strip-fields "heatmap,structure" --eval-batch-size 32 --diagnostic-samples 0 \
  --tag qlora_strip_heatmap_structure

# Paired zero-shot re-run on the unfiltered test split (Together AI, concurrency=8)
TOGETHER_API_KEY=... python -m cli run-backtest \
  --dataset backtest/data/labeled/dataset_no_drawdown_filter.jsonl \
  --provider together --model "Qwen/Qwen2.5-7B-Instruct-Turbo" \
  --tag zero-shot-qwen7b-no-drawdown --split test --concurrency 8

# Paired comparison (not yet run)
python -m cli compare-stats \
  --a optimization/qlora/results/qlora_no_drawdown/result.json \
  --b backtest/data/results/zero-shot-qwen7b-no-drawdown.json
```

---

## Key Files to Check When Resuming

- Filter logic: `langgraph/backtest/ingestion/labeler.py` (`apply_drawdown_filter`)
- Pipeline: `langgraph/backtest/pipeline.py`
- Dataset generation CLI: `langgraph/cli.py` (`cmd_prepare_dataset`)
- Training script + dataset selection + occlusion probes: `langgraph/optimization/qlora/train_qlora.py`
  (`DATASET_TYPES`, `--dataset-type`, `prepare_data`, `evaluate`, `_occlude`, `--strip-fields`, `--anonymize-symbol`)
- Training-data export with occlusion: `langgraph/backtest/export.py` (`build_training_example`, `export_training_data`)
- Baseline result to beat/compare: `optimization/qlora/results/qlora_cloud/result.json` (88.03%)
- Zero-shot baseline: `backtest/data/results/zero-shot-qwen7b.json` (58.49%)
- Results notebook: `langgraph/optimization-results.ipynb` Part 8 (drawdown ablation) and Part 9 (symbol anonymization)
- Audit addendum: `docs/qlora/AUDIT_QLORA_88PCT.md` §10
