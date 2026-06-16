# Session State — No-Drawdown-Filter Experiment

**Preparation date:** 2026-06-16
**Prepared by:** Claude (Opus 4.8), at the user's request
**Branch:** `experiment/no-drawdown-filter` (branched from `dev`)
**Status:** ✅ Prepared — code + docs done. ⏳ **No training executed (by design).**

---

## Code Status

| Component | Status | File / Notes |
|-----------|--------|--------------|
| Labeler — conditional drawdown filter | ✅ | `langgraph/backtest/ingestion/labeler.py` — `apply_drawdown_filter=True` param on `label_candle` + `generate_labeled_dataset` |
| Data pipeline — flag plumbed | ✅ | `langgraph/backtest/pipeline.py` — `DataPipeline(apply_drawdown_filter=...)` |
| CLI — `--no-drawdown-filter` / `--output` | ✅ | `langgraph/cli.py` — `prepare-dataset` |
| Training — `--dataset-type` / `--dataset` | ✅ | `langgraph/optimization/qlora/train_qlora.py` — defaults reproduce original run |
| `run_cloud.sh` | ✅ no change | already forwards `--dataset-type no_filter` to the trainer |
| One-command launcher | ✅ added | `optimization/qlora/run_experiment_no_drawdown_filter.sh` — preflight → dataset → smoke → full run, fail-fast. See EXPERIMENT doc §7 Agent Runbook for the debug tree. |
| Fresh-instance bootstrap | ✅ added | `optimization/qlora/user_data.sh` — EC2 user-data / SageMaker lifecycle style: health checks → AWS CLI → AWS creds (env vars) → `dvc pull` candles → Claude Code → launch. Portable apt/yum. |
| `RUNPOD_GUIDE.md` | ✅ updated | added "Variant: No-Drawdown-Filter Experiment" section (build dataset → run → paired compare) |
| Unit tests | ✅ added + passing | `tests/backtest/test_backtest.py` (drawdown toggle, generate_dataset forwarding, pipeline flag) + `tests/optimization/test_qlora.py` (DATASET_TYPES mapping, config defaults). **81 passed.** |
| Ruff lint + format | ✅ clean | `ruff 0.10.0` installed in the `trading` conda env + added to `langgraph/requirements.txt`. `ruff check` and `ruff format --check` both pass on all changed files. |
| Documentation (`.claude/`) | ✅ | this file + 3 others (see README) |

---

## Measured (2026-06-16)

Candles pulled and labeled both ways. Filtered total reproduces the canonical
**56,161** exactly (validation). Unfiltered = **75,289** → the drawdown filter
removes **19,128 samples (+34.1%)**, uniform across symbols (32–38%). The unfiltered
dataset is on disk at `backtest/data/labeled/dataset_no_drawdown_filter.jsonl` (70 MB,
git-ignored) — `dvc add` it before sharing/training on a remote box.

## Pending Tasks (execution — not yet done)

- [x] `dvc pull backtest/data/candles`
- [x] Generate unfiltered dataset → **75,289 samples (+34.1% vs 56,161)**
- [ ] `dvc add` + push the unfiltered dataset (so the GPU box can pull it)
- [ ] Fine-tune `qlora_no_drawdown_filter` (RunPod A100, ~$6–8, ~4–4.7 h)
- [ ] Re-run zero-shot on the unfiltered test split (for a paired McNemar)
- [ ] `compare-stats` no-filter QLoRA vs no-filter zero-shot
- [ ] DVC-track + push the new model and dataset
- [ ] Apply the `RESULT_INTERPRETATION.md` decision table → update the thesis framing

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
   - model → `backtest/data/models/qlora_no_drawdown_filter/`
4. **Model tag:** `qlora_no_drawdown_filter` everywhere.
5. **Backward compatibility:** all new flags default to the original behavior; running
   the existing commands without new flags is byte-for-byte unchanged.

---

## Quick Commands

```bash
cd langgraph

# 0. ONE-COMMAND launcher (preflight → dataset → smoke → full run, fail-fast):
bash optimization/qlora/run_experiment_no_drawdown_filter.sh --smoke-only   # validate box
nohup bash optimization/qlora/run_experiment_no_drawdown_filter.sh --skip-smoke \
  > optimization/qlora/logs/qlora_no_drawdown_filter.log 2>&1 & echo "PID: $!"
# --- or run the stages manually (below) ---

# 1. (GPU box) candles required for labeling + trade sim
dvc pull backtest/data/candles

# 2. Generate unfiltered dataset (CPU, minutes). Prints exact sample counts.
python -m cli prepare-dataset \
  --symbols BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,DOTUSDT,DOGEUSDT,LINKUSDT,MATICUSDT,NEARUSDT \
  --timeframes "1h,4h,1d" \
  --no-drawdown-filter

# 3. Train (RunPod). Hyperparams match qlora_cloud baseline.
nohup bash optimization/qlora/run_cloud.sh \
  --tag qlora_no_drawdown_filter \
  --lr 0.00002 --rank 16 --alpha 32 --epochs 2 \
  --dataset-type no_filter \
  > optimization/qlora/logs/qlora_no_drawdown_filter.log 2>&1 & echo "PID: $!"

# 4. Eval-only (optional, from saved adapters)
python optimization/qlora/train_qlora.py \
  --eval-only backtest/data/models/qlora_no_drawdown_filter \
  --dataset-type no_filter --eval-batch-size 32 \
  --tag qlora_no_drawdown_filter

# 5. Paired comparison on the unfiltered test set
python -m cli run-backtest \
  --dataset backtest/data/labeled/dataset_no_drawdown_filter.jsonl \
  --provider ollama --split test --tag zero-shot-no-filter
python -m cli compare-stats \
  --a optimization/qlora/results/qlora_no_drawdown_filter/result.json \
  --b backtest/data/results/zero-shot-no-filter.json
```

---

## Key Files to Check When Resuming

- Filter logic: `langgraph/backtest/ingestion/labeler.py` (`apply_drawdown_filter`)
- Pipeline: `langgraph/backtest/pipeline.py`
- Dataset generation CLI: `langgraph/cli.py` (`cmd_prepare_dataset`)
- Training script + dataset selection: `langgraph/optimization/qlora/train_qlora.py`
  (`DATASET_TYPES`, `--dataset-type`, `prepare_data`, `evaluate`)
- Baseline result to beat/compare: `optimization/qlora/results/qlora_cloud/result.json` (88.03%)
- Zero-shot baseline: `backtest/data/results/zero-shot-qwen7b.json` (58.49%)
