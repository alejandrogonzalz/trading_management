# QLoRA Fine-Tuning — Status & Guide

## What We're Doing

Fine-tuning Qwen 2.5 7B (4-bit quantized) to predict crypto trade direction (LONG/SHORT) from multi-timeframe technical indicators. Part of master's thesis comparing fine-tuned LLM vs zero-shot LLM vs ML models (LSTM, XGBoost, SVM).

The model learns to output structured JSON: `{"bias", "entry", "tp", "sl", "reasoning", "confidence"}` given a system prompt + indicator snapshot.

---

## Current State (2026-06-13)

> 🔴 **Leakage fix applied — prior results invalidated.** The 92% config-1 result
> was trained on a positional (per-symbol) split and evaluated on 2000 single-symbol
> (LINK) rows with financial metrics 0. `_temporal_split` is now a strict temporal
> holdout (global timestamp sort + embargo). **Re-train ONE model** on the fixed
> split (`run_cloud.sh`), optionally one local (`run_local.sh`) — **no config sweep**.
> See `docs/AUDITORIA_QLORA_LEAKAGE_OVERFITTING.md` + `docs/GUIA_IMPLEMENTACION_FIX_QLORA.md`.

### Done

| What | Result | Where |
|------|--------|-------|
| Dataset (56K samples) | 39,312 train / 8,424 val / 8,425 test (pre-fix counts) | `backtest/data/labeled/dataset.jsonl` |
| Training data export | Chat-format JSONL | Generated at runtime by `train_qlora.py` |
| SageMaker setup | ml.g6e.xlarge (L40S 48GB), venv, SSH key | Running |
| Leakage fix + instrumentation | strict split, early stopping, gap, loss curve, baseline | `features._temporal_split`, `train_qlora.py` |
| DVC tracking | Dataset, candles, models | Remote: `s3://trading-management-dvc/` |
| ~~Config 1 training (92%)~~ | **INVALID** — contaminated split + partial eval | `optimization/qlora/results/archive/` |

### Pending

| What | Priority | Time estimate |
|------|----------|---------------|
| `dvc pull` dataset + candles (before any eval) | HIGH | minutes |
| Re-train ONE cloud model on fixed split (`run_cloud.sh`) | HIGH | ~3-6h on SageMaker |
| Optional local model on fixed split (`run_local.sh`) | MEDIUM | ~15-19h (1 epoch) |
| Re-train ML/ensembles + re-run `Avance5.ipynb` (Tarea 7) | HIGH | CPU, parallel to GPU |
| Zero-shot LLM backtest on the fixed test | HIGH | ~2-4h with Groq |
| McNemar + t-test comparisons (paired by `sample_keys`) | HIGH | 30 min (once results exist) |
| Integrate best model (GGUF → Ollama → LangGraph) | HIGH | 2-3h |
| Ensemble LSTM+LLM | LOW | 1 day if time permits |

---

## Hyperparameter Config (single model — no sweep)

| Where | LR | Rank | Alpha | Epochs | Batch | Grad Accum | Effective Batch |
|-------|-----|------|-------|--------|-------|------------|-----------------|
| Cloud (`run_cloud.sh`) | 2e-5 | 16 | 32 | 3 | 2 | 8 | 16 |
| Local (`run_local.sh`, optional) | 5e-5 | 8 | 16 | 1 | 1 | 16 | 16 |

The old 3-/5-config sweep was dropped: the thesis needs one defensible model, and
the audit fix invalidated the swept results. `optimization/configs/qlora.yaml` keeps
the other combos as a reference only.

---

## Infrastructure

### EC2 g6e.xlarge + Deep Learning AMI (primary training — recommended)
- GPU: NVIDIA L40S 48GB (Ada `sm_89`), ~$1.86/hr on-demand
- AMI: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04)`
- Venv: `.venv/` (created by `setup_ec2.sh`, NO --system-site-packages)
- **FlashAttention-2 WORKS** (matched CUDA) → ~4-5s/step, ~5-7h for 3 epochs
- Launch with `nohup` (tmux also available on raw EC2). STOP the instance when done.
- Guide: `optimization/qlora/EC2_GUIDE.md` (includes EC2-vs-SageMaker cost analysis)

### SageMaker Studio (fallback only)
- ml.g6e.xlarge L40S 48GB, ~$2.00-2.35/hr — same `setup_ec2.sh` works here
- No FA2 (container CUDA mismatch) → ~8.7s/step (~2x slower), no tmux → nohup
- Use only if EC2 quota/access is blocked; EC2 is cheaper and faster

### Local (RTX 5070 Ti 16GB, Windows)
- Path: `C:\Users\alex\projects\trading_management\langgraph`
- Venv: `.venv-finetuning/`
- batch_size=1 only (16GB limit), ~28s/step
- No FA2 on Windows — Unsloth uses Triton kernels instead
- Guide: `trading_management_docs/fine-tunning/LOCAL_TRAINING_RTX5070Ti.md`

---

## Key Constraints (learned from errors)

1. **max_seq_length=1024** — samples are 877-933 tokens. Lower = crash on step 0 (Unsloth padding-free bug)
2. **batch_size=2 max on L40S** — batch=4 OOMs during backward pass
3. **Always nvidia-smi before launching** — zombie processes hold 35GB from failed runs
4. **per_device_eval_batch_size=1** — HF default of 8 OOMs with 900-token samples
5. **No --system-site-packages** — SageMaker base has TF/Keras3 that breaks transformers
6. **Candles required for trade simulation** — without them, win_rate=0.0 (DVC-tracked files)
7. **TRL version detection** — script auto-detects SFTConfig (>=0.12) vs legacy API
8. **Strict temporal split** — `_temporal_split` sorts globally by timestamp + embargo.
   NEVER cut by file position (`dataset.jsonl` is grouped by symbol → per-symbol leak).
9. **Full test eval, no `--max-eval`** — the cap is what limited the old eval to one
   symbol (LINK). Leave it off for real runs; only use it for smoke tests.
10. **Overfitting diagnostics emitted** — real `EarlyStoppingCallback` (patience 3),
    `overfitting`{train/val/test acc + gap}, `baseline_metrics` (heuristic), and a
    `<tag>_loss_curve.{json,png}`. Read the gap, not just accuracy.

---

## Key Files

| File | Purpose |
|------|---------|
| `optimization/qlora/train_qlora.py` | Main training + evaluation script |
| `optimization/qlora/run_cloud.sh` | Single cloud run (SageMaker) on the fixed split |
| `optimization/qlora/run_local.sh` | Single local run (RTX 5070 Ti), optional |
| `optimization/qlora/setup_ec2.sh` | GPU-instance env setup (EC2 DLAMI; works on SageMaker too) |
| `optimization/qlora/EC2_GUIDE.md` | Step-by-step EC2 walkthrough + cost analysis |
| `optimization/qlora/results/qlora_<tag>.json` | Result JSON (+ canonical `qlora_optimization.json`) |
| `optimization/qlora/results/archive/` | Invalid old sweep results (do NOT cite) |
| `optimization/qlora/logs/` | Training logs per run |
| `backtest/data/models/<tag>/` | Adapters + GGUF (DVC-tracked) |
| `backtest/export.py` | Converts dataset → chat-format JSONL |
| `agent/prompts.py` | System/user prompts (shared with training) |
| `.claude/rules/qlora-training.md` | Full hyperparameter reference |
| `trading_management_docs/fine-tunning/` | FAQ, plan, local guide, briefing |

---

## Commands Cheat Sheet

```bash
# SageMaker — REQUIRED before any run (candles → real financial metrics)
dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles

# SageMaker — single cloud run on the fixed split (no --max-eval = full test)
nohup bash optimization/qlora/run_cloud.sh > optimization/qlora/logs/run_cloud.log 2>&1 & echo "PID: $!"

# SageMaker — monitor
tail -f optimization/qlora/logs/run_cloud.log
nvidia-smi
ps aux | grep train_qlora | grep -v grep

# SageMaker — backup to S3
aws s3 cp --recursive backtest/data/models/qlora_cloud/ s3://trading-management-dvc/models/qlora_cloud/

# Local — smoke test (caps eval; only for testing the loop)
.\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py --max-steps 3 --max-eval 10

# Compare models (canonical result file, paired by sample_keys)
python -m cli compare-stats --a optimization/qlora/results/qlora_optimization.json --b backtest/data/results/ml-lstm.json

# Integration (after the re-trained model is validated)
ollama create trading-qwen-ft -f backtest/data/models/qlora_cloud/Modelfile
# Then set LLM_MODEL=trading-qwen-ft in .env
```

---

## Results So Far

### ~~Config 1 (lr=2e-5, rank=16, epochs=2) — 92%~~ → INVALID
Trained on the contaminated positional split and evaluated on 2000 single-symbol
(LINK) rows with financial metrics 0. **Not a valid generalization measure** —
archived under `results/archive/`. The re-trained model on the fixed split replaces it.

### Comparison targets (all need RE-MEASURING on the fixed split — Tarea 7)
- Bagging-LSTM: 83.37% test acc (former ML winner, 5 bags, AUC=0.9157)
- LSTM individual: 81.5% test acc
- Blending ensemble: 81.89% test acc
- XGBoost: 66.6% test acc
- Zero-shot LLM: not yet measured (need to run backtest on the fixed test)

---

## What Success Looks Like

1. Fine-tuned model beats zero-shot LLM in accuracy **on the fixed temporal test**
   (and clearly beats the heuristic baseline, not just majority-class 51%)
2. Fine-tuned model has profit_factor > 1.0 in backtest (needs candles via `dvc pull`)
3. Small train−test **gap** + non-diverging loss curve → no overfitting signal
4. Statistical significance confirmed via McNemar test (p < 0.05), paired by `sample_keys`
5. Model integrated in LangGraph agent (GGUF → Ollama → `LLM_MODEL` env var)

---

## Timeline to Thesis Defense (~Jun 26)

| Days | What |
|------|------|
| Jun 13-14 | `dvc pull`; re-train ONE cloud model on fixed split; re-run ML/ensembles (Tarea 7) |
| Jun 14-15 | Optional local model (overnight), zero-shot backtest on fixed test |
| Jun 15-16 | All results in hand, run statistical comparisons (paired) |
| Jun 16-17 | LangGraph integration, screenshots/video |
| Jun 19-22 | Documentation, report, presentation |
| Jun 23-25 | Buffer + rehearsal |
| ~Jun 26 | Defense |
