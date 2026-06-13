# QLoRA Fine-Tuning — Status & Guide

## What We're Doing

Fine-tuning Qwen 2.5 7B (4-bit quantized) to predict crypto trade direction (LONG/SHORT) from multi-timeframe technical indicators. Part of master's thesis comparing fine-tuned LLM vs zero-shot LLM vs ML models (LSTM, XGBoost, SVM).

The model learns to output structured JSON: `{"bias", "entry", "tp", "sl", "reasoning", "confidence"}` given a system prompt + indicator snapshot.

---

## Current State (2026-06-13)

### Done

| What | Result | Where |
|------|--------|-------|
| Dataset (56K samples) | 39,312 train / 8,424 val / 8,425 test | `backtest/data/labeled/dataset.jsonl` |
| Training data export | Chat-format JSONL | Generated at runtime by `train_qlora.py` |
| SageMaker setup | ml.g6e.xlarge (L40S 48GB), venv, SSH key | Running |
| Config 1 training | 92% direction accuracy (2000 test samples) | `optimization/qlora/results/qlora_config1.json` |
| Model backup | Adapters + GGUF in S3 | `s3://trading-management-dvc/models/qlora_config1/` |
| DVC tracking | Dataset, candles, models | Remote: `s3://trading-management-dvc/` |

### In Progress

| What | Status | Notes |
|------|--------|-------|
| Config 1 eval WITH candles | Running on SageMaker (nohup) | Re-running to get win_rate, profit_factor, Sharpe |

### Pending

| What | Priority | Time estimate |
|------|----------|---------------|
| Config 2 (lr=5e-5, rank=8, epochs=3) | HIGH | ~12-14h on SageMaker |
| Config 3 (lr=1e-5, rank=32, epochs=2) | MEDIUM | ~14-16h on SageMaker |
| Local training (RTX 5070 Ti, rank=8, 1 epoch) | MEDIUM | ~19h |
| Zero-shot LLM backtest | HIGH | ~2-4h with Groq |
| McNemar + t-test comparisons | HIGH | 30 min (once results exist) |
| Integrate best model (GGUF → Ollama → LangGraph) | HIGH | 2-3h |
| Ensemble LSTM+LLM | LOW | 1 day if time permits |

---

## Hyperparameter Configs

| Config | LR | Rank | Alpha | Epochs | Batch | Grad Accum | Effective Batch |
|--------|-----|------|-------|--------|-------|------------|-----------------|
| 1 (done) | 2e-5 | 16 | 32 | 2 | 2 | 8 | 16 |
| 2 (pending) | 5e-5 | 8 | 16 | 3 | 2 | 8 | 16 |
| 3 (pending) | 1e-5 | 32 | 64 | 2 | 2 | 8 | 16 |
| Local (pending) | 5e-5 | 8 | 16 | 1 | 1 | 16 | 16 |

---

## Infrastructure

### SageMaker (primary training)
- Instance: ml.g6e.xlarge — L40S 48GB (44GB usable), ~$2.35/hr
- Path: `/home/sagemaker-user/trading_management/langgraph/`
- Venv: `.venv/` (created by `sagemaker_setup.sh`, NO --system-site-packages)
- No FlashAttention-2 (CUDA mismatch), no tmux — use nohup
- ~8.7s/step (vs ~3-4s expected with FA2)
- Git remote: SSH (ed25519 key configured)

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

---

## Key Files

| File | Purpose |
|------|---------|
| `optimization/qlora/train_qlora.py` | Main training + evaluation script |
| `optimization/qlora/run_3configs.sh` | Sequential wrapper for 3 configs |
| `optimization/qlora/sagemaker_setup.sh` | SageMaker env setup |
| `optimization/qlora/results/qlora_config*.json` | Result JSONs |
| `optimization/qlora/logs/` | Training logs per config |
| `backtest/data/models/qlora_config*/` | Adapters + GGUF (DVC-tracked) |
| `backtest/export.py` | Converts dataset → chat-format JSONL |
| `agent/prompts.py` | System/user prompts (shared with training) |
| `.claude/rules/qlora-training.md` | Full hyperparameter reference |
| `trading_management_docs/fine-tunning/` | FAQ, plan, local guide, briefing |

---

## Commands Cheat Sheet

```bash
# SageMaker — monitor current run
tail -20 optimization/qlora/logs/eval_config1.log
nvidia-smi
ps aux | grep train_qlora | grep -v grep

# SageMaker — launch config 2
nohup python optimization/qlora/train_qlora.py \
  --lr 0.00005 --rank 8 --alpha 16 --epochs 3 \
  --batch-size 2 --grad-accum 8 --max-eval 2000 \
  --output-dir backtest/data/models/qlora_config2 \
  > optimization/qlora/logs/qlora_config2.log 2>&1 & echo "PID: $!"

# SageMaker — backup to S3
aws s3 cp --recursive backtest/data/models/qlora_config2/ s3://trading-management-dvc/models/qlora_config2/

# Local — smoke test
.\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py --max-steps 3 --max-eval 10

# Compare models (after all results exist)
python -m cli compare-stats --a optimization/qlora/results/qlora_config1.json --b backtest/data/results/ml-lstm.json

# Integration (after picking best model)
ollama create trading-qwen-ft -f backtest/data/models/qlora_config1/Modelfile
# Then set LLM_MODEL=trading-qwen-ft in .env
```

---

## Results So Far

### Config 1 (lr=2e-5, rank=16, epochs=2)
- Direction accuracy: **92%** (on 2000 test samples)
- Trade metrics: **PENDING** (re-running eval with candles)
- Training time: ~14h on SageMaker (~$33)
- Trainable params: 40M / 4.3B (0.9%)

### Comparison targets
- Bagging-LSTM: 83.37% test acc (current ML winner, 5 bags, AUC=0.9157)
- LSTM individual: 81.5% test acc
- Blending ensemble: 81.89% test acc
- XGBoost: 66.6% test acc
- Zero-shot LLM: not yet measured (need to run backtest)

---

## What Success Looks Like

1. Fine-tuned model beats zero-shot LLM in accuracy (very likely given 92% vs expected ~50-60%)
2. Fine-tuned model has profit_factor > 1.0 in backtest (needs trade metrics)
3. Statistical significance confirmed via McNemar test (p < 0.05)
4. Model integrated in LangGraph agent (GGUF → Ollama → `LLM_MODEL` env var)
5. At least 2 configs compared to show hyperparameter sensitivity

---

## Timeline to Thesis Defense (~Jun 26)

| Days | What |
|------|------|
| Jun 13-14 | Eval with candles finishes, launch configs 2-3 |
| Jun 14-15 | Local training (overnight), zero-shot backtest |
| Jun 15-16 | All results in hand, run statistical comparisons |
| Jun 16-17 | LangGraph integration, screenshots/video |
| Jun 19-22 | Documentation, report, presentation |
| Jun 23-25 | Buffer + rehearsal |
| ~Jun 26 | Defense |
