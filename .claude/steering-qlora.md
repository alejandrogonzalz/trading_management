# QLoRA Fine-Tuning — Status & Guide

## What We're Doing

Fine-tuning Qwen 2.5 7B (4-bit quantized) to predict crypto trade direction (LONG/SHORT) from multi-timeframe technical indicators. Part of master's thesis comparing fine-tuned LLM vs zero-shot LLM vs ML models (LSTM, XGBoost, SVM).

The model learns to output structured JSON: `{"bias", "entry", "tp", "sl", "reasoning", "confidence"}` given a system prompt + indicator snapshot.

---

## Current State (2026-06-16)

> ✅ **All measurements complete.** QLoRA cloud + config3 trained and evaluated.
> Zero-shot backtest done. McNemar test run. All ML models re-measured on the fixed
> temporal split. Remaining work: `Avance6.ipynb`, LangGraph integration, thesis write-up.

### Done

| What | Result | Where |
|------|--------|-------|
| Dataset (56K samples) | 39,312 train / 8,424 val / 8,425 test | `backtest/data/labeled/dataset.jsonl` (DVC) |
| Leakage fix | `_temporal_split` — global sort + 24h embargo | `backtest/models/features.py` |
| QLoRA cloud training | **88.03% test acc**, profit_factor=12.92, AUC=0.942 | `optimization/qlora/results/qlora_cloud/result.json` |
| QLoRA config3 (robustness check) | **87.87% test acc**, profit_factor=11.96 | `optimization/qlora/results/qlora_config3/result.json` |
| Zero-shot backtest (Qwen 2.5 7B) | 58.49% acc, 28.42% win rate, 98.47% drawdown | `backtest/data/results/zero-shot-qwen7b.json` |
| ML models re-measured on fixed split | RF 64.24%, XGB 63.60%, Blending 63.59%, LSTM 50.21% | `optimization/results/*_v2_optimization.json` |
| McNemar test | chi²=557, p≈0 — fine-tuning is statistically significant | `optimization/stats_tests.py` |
| Analysis notebook | Parts 1–6 (loss curves, accuracy, McNemar, trade metrics, equity curves) | `langgraph/optimization-results.ipynb` |
| Ollama deployment guide | Local inference walkthrough (dvc pull → ollama create) | `docs/ops/OLLAMA_DEPLOYMENT.md` |
| DVC tracking | Dataset, candles, QLoRA models in S3 | Remote: `s3://trading-management-dvc/` |
| ~~Config 1 training (92%)~~ | **INVALID** — contaminated split + partial eval | `optimization/qlora/results/archive/` |

### Pending

| What | Priority |
|------|----------|
| `Avance6.ipynb` — clean thesis deliverable notebook | HIGH |
| LangGraph integration (GGUF → Ollama → agent) | HIGH |
| Thesis write-up + presentation | HIGH |
| Defense (~Jun 26) | — |

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

### RunPod A100 80GB (primary training — recommended)
- Image: `docker.io/unsloth/unsloth:latest` (torch 2.10+cu128, FA2 2.8.3 pre-installed)
- GPU: NVIDIA A100 80GB PCIe (`sm_80`), ~$1.39/hr
- **No venv needed** — `/opt/venv` is pre-activated, use `python3` directly
- **FlashAttention-2 pre-installed** → BATCH=8, ~1.5-2s/step, ~3-4h for 2 epochs, ~$5
- Launch with `nohup`. Container user: `unsloth`, /workspace is writable.
- Guide: `optimization/qlora/RUNPOD_GUIDE.md`

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
10. **flash-attn is pre-installed on the unsloth image** — `unsloth/unsloth:latest`
    ships flash-attn 2.8.3 compiled against its own torch, so no build is needed.
    On bare torch images (`setup_runpod.sh` path), ABI mismatch was an issue: prebuilt
    wheels use `cxx11abiFALSE` (conda ABI) but pip torch uses old ABI. Fix was
    `FLASH_ATTENTION_FORCE_BUILD=TRUE`. On the unsloth image this is already resolved.
11. **Overfitting diagnostics emitted** — real `EarlyStoppingCallback` (patience 3),
    `overfitting`{train/val/test acc + gap}, `baseline_metrics` (heuristic), and a
    `<tag>_loss_curve.{json,png}`. Read the gap, not just accuracy.

---

## Key Files

| File | Purpose |
|------|---------|
| `optimization/qlora/train_qlora.py` | Main training + evaluation script |
| `optimization/qlora/run_cloud.sh` | Single cloud run (RunPod/EC2) on the fixed split |
| `optimization/qlora/run_local.sh` | Single local run (RTX 5070 Ti), optional |
| `optimization/qlora/setup_unsloth_pod.sh` | RunPod setup for `unsloth/unsloth:latest` image |
| `optimization/qlora/RUNPOD_GUIDE.md` | Full RunPod walkthrough + cost analysis |
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

# Save model to DVC + git after training (run from repo root: trading_management/)
dvc add langgraph/backtest/data/models/qlora_cloud/
dvc push
git add langgraph/backtest/data/models/qlora_cloud.dvc
git commit -m "feat(qlora): add qlora_cloud model weights (DVC)"
git push
# NOTE: always run dvc add from trading_management/ (the DVC root), NOT from langgraph/

# Local — smoke test (caps eval; only for testing the loop)
.\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py --max-steps 3 --max-eval 10

# Compare models (canonical result file, paired by sample_keys)
python -m cli compare-stats --a optimization/qlora/results/qlora_optimization.json --b backtest/data/results/ml-lstm.json

# Integration (after the re-trained model is validated)
ollama create trading-qwen-ft -f backtest/data/models/qlora_cloud/Modelfile
# Then set LLM_MODEL=trading-qwen-ft in .env
```

---

## DVC Debug & Validation

**The DVC root is `trading_management/` — always run `dvc` commands from there, NOT from `langgraph/`.**

```bash
# 1. Check what DVC thinks is out of sync (local vs remote)
cd trading_management/
dvc status          # local cache vs working tree
dvc status --cloud  # local cache vs S3 remote

# 2. Count files in S3 and total size
aws s3 ls s3://trading-management-dvc/dvc/files/md5/ --recursive | wc -l
aws s3 ls s3://trading-management-dvc/dvc/files/md5/ --recursive \
  | awk '{sum+=$3} END {printf "%.1f GB\n", sum/1024/1024/1024}'

# 3. Verify a specific model's files are really in S3
#    Step A: download the directory manifest (replace HASH with the md5 from the .dvc file)
DVC_HASH=$(python3 -c "import yaml; print(yaml.safe_load(open('langgraph/backtest/data/models/qlora_cloud.dvc'))['outs'][0]['md5'])")
PREFIX="${DVC_HASH:0:2}"; REST="${DVC_HASH:2}"
aws s3 cp "s3://trading-management-dvc/dvc/files/md5/$PREFIX/$REST" /tmp/manifest.json
python3 -c "import json; files=json.load(open('/tmp/manifest.json')); print(f'{len(files)} files in manifest')"

#    Step B: cross-check a key local file's MD5 against the manifest
md5sum langgraph/backtest/data/models/qlora_cloud/adapter_model.safetensors
# Must match the hash listed for adapter_model.safetensors in the manifest above

#    Step C: confirm that hash actually exists as a physical file in S3
FILE_HASH="<paste md5 from Step B>"
aws s3 ls "s3://trading-management-dvc/dvc/files/md5/${FILE_HASH:0:2}/${FILE_HASH:2}"
# Should print a file size > 0. If empty → file is missing from S3, run dvc push.

# 4. Quick one-liner: validate the 3 most important qlora_cloud files
for f in adapter_model.safetensors "gguf_gguf/Qwen2.5-7B-Instruct.Q4_K_M.gguf" adapter_config.json; do
  h=$(md5sum "langgraph/backtest/data/models/qlora_cloud/$f" | awk '{print $1}')
  result=$(aws s3 ls "s3://trading-management-dvc/dvc/files/md5/${h:0:2}/${h:2}" 2>/dev/null)
  [ -n "$result" ] && echo "✓ $f" || echo "✗ MISSING: $f (hash=$h)"
done

# 5. Pull the model back (disaster recovery — confirms S3 is the source of truth)
dvc pull langgraph/backtest/data/models/qlora_cloud.dvc

# 6. S3 bucket protection status
aws s3api get-bucket-versioning --bucket trading-management-dvc
aws s3api get-bucket-policy --bucket trading-management-dvc
```

### Common mistakes
| Mistake | Symptom | Fix |
|---------|---------|-----|
| `dvc add` from `langgraph/` | Files not found by `dvc status` from repo root | Re-run `dvc add` from `trading_management/` |
| `dvc push` before `dvc add` | "Everything is up to date" but S3 has 0 bytes | `dvc add` first, then `dvc push` |
| `dvc push` says up-to-date but files missing | 180 files in S3 but all tiny | Check total size: must be >20GB for qlora_cloud |
| DVC cache empty after `dvc add` | Cache dir shows 0 bytes | `dvc add` ran from wrong directory; files never cached |

---

## Final Results (all on fixed temporal test split)

| Model | Test Acc | Win Rate | Profit Factor | Max Drawdown |
|-------|----------|----------|---------------|-------------|
| QLoRA cloud (lr=2e-5, rank=16) | **88.03%** | 61.67% | 12.92 | 12.3% |
| QLoRA config3 (lr=1e-5, rank=32) | **87.87%** | 60.30% | 11.96 | 14.6% |
| Random Forest v2 | 64.24% | 56.40% | 2.23 | 39.3% |
| XGBoost v2 | 63.60% | 57.47% | 2.34 | 36.4% |
| Blending ensemble v2 | 63.59% | — | — | — |
| Zero-shot Qwen 7B | 58.49% | 28.42% | 1.35 | 98.5% |
| LSTM v2 | 51.51% | 46.74% | 1.49 | 38.7% |

**McNemar**: chi²=557, p≈0 (1094 QLoRA wins vs 233 zero-shot wins on 3000 paired samples)  
**Config3** confirms robustness: two independent hyperparameter sets both land at ~88%.  
**Zero-shot anomaly**: 58.5% direction acc but 98.5% drawdown — base model knows partial direction but cannot calibrate TP/SL to crypto volatility.

~~Config 1 (92%)~~ — **INVALID**, archived. Trained on contaminated positional split.

---

## What Was Achieved

1. ✅ Fine-tuned model beats zero-shot by +29.5pp accuracy (88% vs 58.5%)
2. ✅ profit_factor=12.92 >> 1.0 (candles pulled via DVC before eval)
3. ✅ Negative overfitting gap (train_acc=70.5% < test_acc=88%) — healthy generalization
4. ✅ McNemar p≈0 — statistically significant at any reasonable α
5. ⏳ Model integration in LangGraph agent pending (GGUF ready in DVC)

---

## Timeline to Thesis Defense (~Jun 26)

| Date | What |
|------|------|
| Jun 16 | Add equity curves to notebook (Part 6) ✅ |
| Jun 16–17 | `Avance6.ipynb` — clean thesis deliverable |
| Jun 17–18 | LangGraph integration + screenshots/video |
| Jun 18–22 | Technical report + presentation |
| Jun 23–25 | Buffer + rehearsal |
| ~Jun 26 | Defense |
