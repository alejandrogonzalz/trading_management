# QLoRA Fine-Tuning on RunPod

Step-by-step guide to fine-tune Qwen 2.5 7B on a **RunPod A100 80GB** using the
official `unsloth/unsloth:latest` Docker image.

> **Validated 2026-06-15 on A100 80GB PCIe.**
> Image ships torch 2.10.0+cu128, flash-attn 2.8.3, unsloth, bitsandbytes, TRL — no
> build step needed. `setup_unsloth_pod.sh` validates the stack, installs the few
> missing project deps (DVC, sklearn, xgboost), and runs a smoke test.

---

## Pod Configuration

| Field | Value |
|-------|-------|
| **Image** | `docker.io/unsloth/unsloth:latest` |
| **GPU** | A100 80GB PCIe (recommended) — H100 SXM also works |
| **Container Disk** | **100 GB** (image ~25 GB + model weights + checkpoints) |
| **Volume Disk** | 100–256 GB mounted at `/workspace` |
| **HTTP Port** | 8888 (JupyterLab, optional) |
| **Container user** | `unsloth` (no root needed, /workspace is writable) |

---

## What's Pre-Installed (confirmed)

| Component | Version |
|-----------|---------|
| Python | 3.12.3 |
| torch | 2.10.0+cu128 |
| CUDA toolkit | 12.8 (nvcc present) |
| unsloth | 2026.5.x |
| flash-attn | **2.8.3 pre-installed** ← no build needed |
| bitsandbytes | 0.49.x |
| TRL | 0.23.x |
| transformers | 4.57.x |
| peft | 0.18.x |
| Active venv | `/opt/venv` (auto-activated, no `source` needed) |

**No virtual environment activation required.** `python3` already points to `/opt/venv/bin/python3`.

---

## Step 1: Clone the Repo

```bash
cd /workspace/work
git clone https://github.com/luisaga215/trading_management.git
cd trading_management/langgraph
```

If the repo is already there:

```bash
cd /workspace/work/trading_management
git pull
cd langgraph
```

---

## Step 2: Run Setup (~2–5 min)

```bash
bash optimization/qlora/setup_unsloth_pod.sh
```

Validates the pre-installed stack, installs missing project deps (DVC, sklearn,
xgboost), installs AWS CLI, and runs a 3-step smoke test to verify everything works.

Expected output:
```
[1/7] GPU validation...    → A100 80GB PCIe, 80 GB, CUDA 12.8
[2/7] Validating stack...  → unsloth OK, flash-attn 2.8.3, bitsandbytes 0.49.x
[3/7] Dependencies...      → installs dvc[s3], xgboost (others already present)
[4/7] FlashAttention-2...  → Already installed: v2.8.3
[5/7] AWS CLI...           → installs if missing
[6/7] DVC data...          → Skipped (configure AWS first)
[7/7] Smoke test...        → 3 training steps pass
Recommended: BATCH=8 GRAD_ACCUM=2 → ~1.5-2s/step (~2-3h total)
```

---

## Step 3: Configure AWS + Pull Data

**REQUIRED before training.** Without candles, trade simulation metrics
(win_rate, profit_factor, Sharpe) come out as 0.

```bash
aws configure
# Enter: Access Key, Secret Key, Region: us-east-1, Output: json

# Pull dataset and candles
dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles

# Verify
wc -l backtest/data/labeled/dataset.jsonl  # must print: 56161
ls backtest/data/candles/ | wc -l          # must print: 36
```

---

## Step 4: Smoke Test

Run a quick 3-step test to confirm the full pipeline (load model → train → eval → GGUF).

### Option A: Foreground (output in terminal)

```bash
python3 optimization/qlora/train_qlora.py \
  --max-steps 3 --max-eval 4 --eval-batch-size 4 \
  --diagnostic-samples 0 --batch-size 4 --no-gguf --tag smoke_full
```

### Option B: Background + log file (recommended if on JupyterLab or closing terminal)

```bash
python3 optimization/qlora/train_qlora.py \
  --max-steps 3 --max-eval 4 --eval-batch-size 4 \
  --diagnostic-samples 0 --batch-size 4 --no-gguf --tag smoke_full \
  > optimization/qlora/logs/smoke_full.log 2>&1 &
echo "PID: $!"

# Monitor
tail -f optimization/qlora/logs/smoke_full.log
```

### What to look for

| Signal | Expected | Problem if... |
|--------|----------|--------------|
| `FA [... FA2 = True]` | Flash-attn active | `FA2 = False` → xformers fallback (slower, still works) |
| `step 1/3 ... step 3/3` | 3 training steps in ~60s | Crash on step 0 = OOM or import error |
| Eval completes (1 batch of 4) | ~90s for generation warmup, then done | `--eval-batch-size 1` is slow; always use ≥4 for smoke |
| No OOM | No CUDA memory errors | Retry with `--batch-size 2` |

If batch-size 4 OOMs, retry with `--batch-size 2`. If that also OOMs, check `nvidia-smi`
for zombie processes from a failed run.

---

## Step 5: Full Training Run

Once the smoke test passes, launch the real run. This uses `run_cloud.sh` which
auto-selects BATCH=8, GRAD_ACCUM=2 on an A100 80GB with FA2.

```bash
mkdir -p optimization/qlora/logs

nohup bash optimization/qlora/run_cloud.sh \
  > optimization/qlora/logs/run_cloud.log 2>&1 &
echo "Training PID: $!"
```

### Monitor training

```bash
# Live log
tail -f optimization/qlora/logs/run_cloud.log

# GPU utilization (open a second terminal)
watch -n 10 nvidia-smi

# Check which batch/FA2 was auto-selected (first lines of log)
grep -E "(BATCH|FA|Recommended)" optimization/qlora/logs/run_cloud.log | head -10

# Check current loss (most recent training step)
grep "loss" optimization/qlora/logs/run_cloud.log | tail -5
```

### Expected behavior

| Phase | Duration | What you see |
|-------|----------|-------------|
| Model load | ~2 min | HuggingFace download or cache hit |
| Data prep | ~1 min | Temporal split, tokenization |
| Training (2 epochs) | ~3–3.5h | Loss drops ~1.8 → ~0.5, checkpoints every 250 steps |
| Save adapters + GGUF | ~15 min | Merging weights, Q4_K_M quantization |
| Evaluation (3000 strided) | ~10–15 min | Batched greedy decode, trade simulation |
| **Total** | **~4h** | **~$5–6 on A100 @ $1.39/hr** |

### If interrupted

```bash
# Resume from last checkpoint (no work lost):
nohup bash optimization/qlora/run_cloud.sh --resume \
  > optimization/qlora/logs/run_cloud_resume.log 2>&1 &
echo "Resume PID: $!"
tail -f optimization/qlora/logs/run_cloud_resume.log
```

### Running multiple configs for comparison

`run_cloud.sh` accepts `--tag`, `--lr`, `--rank`, `--alpha`, and `--epochs` flags.
Each run produces its own result JSON and model directory, so they don't overwrite
each other. Run sequentially after your first model finishes:

```bash
# Config 1 — default (safe bet, already running or done)
bash optimization/qlora/run_cloud.sh --tag config1

# Config 2 — aggressive (higher lr, smaller adapters, more epochs)
bash optimization/qlora/run_cloud.sh --tag config2 --lr 0.00005 --rank 8 --epochs 3

# Config 3 — conservative (lower lr, larger adapters)
bash optimization/qlora/run_cloud.sh --tag config3 --lr 0.00001 --rank 32 --epochs 2
```

| Config | LR | Rank | Alpha | Epochs | Hypothesis |
|--------|-----|------|-------|--------|-----------|
| config1 | 2e-5 | 16 | 32 | 2 | Literature default — works in 90% of cases |
| config2 | 5e-5 | 8 | 16 | 3 | Task is easy — small adapter + more passes suffice |
| config3 | 1e-5 | 32 | 64 | 2 | Task is complex — needs more adapter capacity |

Each produces:
- `optimization/qlora/results/qlora_<tag>.json` — metrics + per-sample predictions
- `backtest/data/models/<tag>/` — adapters + GGUF

Compare results afterwards:
```bash
python -m cli compare-stats \
  --a optimization/qlora/results/qlora_config1.json \
  --b optimization/qlora/results/qlora_config2.json

python -m cli compare-stats \
  --a optimization/qlora/results/qlora_config1.json \
  --b optimization/qlora/results/qlora_config3.json
```

Any extra flags (e.g. `--resume`, `--max-steps 3`) are forwarded to `train_qlora.py`.
Env vars (`BATCH`, `GRAD_ACCUM`, `MAX_EVAL`, `EVAL_BATCH`) still work for GPU tuning.

---

## Step 6: ML Grid Search in Parallel (CPU only)

Launch while QLoRA trains — all three blocks run on CPU, don't interfere with the GPU job.

### Base models (via `optimize.py` + YAML configs)

```bash
mkdir -p optimization/logs

# XGBoost — RandomizedSearchCV, L1/L2 regularization
OMP_NUM_THREADS=4 python3 optimization/optimize.py \
  --model xgboost --config optimization/configs/xgboost.yaml \
  > optimization/logs/xgboost.log 2>&1 &
echo "XGBoost PID: $!"

# Random Forest — 100 random iterations
OMP_NUM_THREADS=4 python3 optimization/optimize.py \
  --model random_forest --config optimization/configs/random_forest.yaml \
  > optimization/logs/rf.log 2>&1 &
echo "RF PID: $!"

# LSTM — 30 random iterations (force CPU so it doesn't fight QLoRA)
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python3 optimization/optimize.py \
  --model lstm --config optimization/configs/lstm.yaml \
  > optimization/logs/lstm.log 2>&1 &
echo "LSTM PID: $!"
```

### Ensemble models (via `run_ensembles.py`)

Trains BaggingLSTM (5 bags), AdaBoost, SoftVoting (LSTM+XGB), Stacking, and
Blending sequentially. Runs after base models complete (needs their trained weights),
or run independently — it trains its own sub-models internally.

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 python3 optimization/run_ensembles.py \
  > optimization/logs/ensembles.log 2>&1 &
echo "Ensembles PID: $!"
```

### Monitor any job

```bash
tail -f optimization/logs/xgboost.log
tail -f optimization/logs/rf.log
tail -f optimization/logs/lstm.log
tail -f optimization/logs/ensembles.log

# Check which jobs are still running
ps aux | grep -E "optimize|run_ensembles" | grep -v grep
```

### Compare all results when done

```bash
python3 optimization/analyze_results.py
```

---

## Step 7: After Training Completes

### Verify results

```bash
# Check result JSON exists and has content
ls -lh optimization/qlora/results/qlora_cloud.json
python3 -c "
import json
r = json.load(open('optimization/qlora/results/qlora_cloud.json'))
m = r.get('metrics', {})
print(f'Accuracy: {m.get(\"direction_accuracy\", 0):.4f}')
print(f'Win rate: {m.get(\"win_rate\", 0):.4f}')
print(f'Profit factor: {m.get(\"profit_factor\", 0):.4f}')
ov = r.get('overfitting', {})
print(f'Train acc: {ov.get(\"train_acc\", \"N/A\")}')
print(f'Test acc:  {ov.get(\"test_acc\", \"N/A\")}')
print(f'Gap:       {ov.get(\"gap\", \"N/A\")}')
"
```

### Plot overfitting dashboard

```bash
python3 optimization/qlora/plot_overfitting.py \
  --result optimization/qlora/results/qlora_cloud.json
# → results/qlora_cloud_overfitting.png
```

### Compare against LSTM and XGBoost

```bash
python3 -m cli compare-stats \
  --a optimization/qlora/results/qlora_optimization.json \
  --b backtest/data/results/ml-lstm.json
```

---

## Step 8: Backup to S3

Do this before stopping the pod — container disk is ephemeral.

```bash
# Model adapters + GGUF
aws s3 cp --recursive backtest/data/models/qlora_cloud/ \
  s3://trading-management-dvc/models/qlora_cloud/

# Result JSON
aws s3 cp optimization/qlora/results/qlora_cloud.json \
  s3://trading-management-dvc/results/qlora_cloud.json

# Loss curve
aws s3 cp optimization/qlora/results/qlora_cloud_loss_curve.json \
  s3://trading-management-dvc/results/qlora_cloud_loss_curve.json
```

### Commit DVC pointer + results

```bash
dvc add backtest/data/models/qlora_cloud
dvc push

git add backtest/data/models/qlora_cloud.dvc \
        backtest/data/models/.gitignore \
        optimization/qlora/results/qlora_cloud.json \
        optimization/qlora/results/qlora_cloud_loss_curve.json

git commit -m "feat(qlora): fine-tuning results on fixed temporal split (RunPod A100 80GB)"
git push origin fix/qlora-data-leak-v3
```

---

## Cost Comparison by GPU

| GPU | VRAM | $/hr | BATCH/GA | FA2 | Train (2ep) | Total | Cost |
|-----|------|------|----------|-----|------------|-------|------|
| **A100 80GB ⭐** | 80 GB | $1.39 | 8/2 | ✅ | ~3–3.5h | **~4h** | **~$5–6** |
| H100 SXM | 80 GB | $3.29 | 8/2 | ✅ | ~1.5–2.5h | ~2.5h | ~$8 |
| L40S 48GB | 48 GB | $0.86 | 4/4 | ✅ | ~6–7h | ~7h | ~$6 |
| RTX 4090 | 24 GB | $0.69 | 2/8 | ✅ | ~5–6h | ~6h | ~$4 |
| RTX A6000 | 48 GB | $0.49 | 4/4 | ✅ | ~12–14h | ~14h | ~$7 |

For **3 epochs** instead of 2, multiply the training column by ~1.5.

> **Avoid Blackwell (B200/RTX 5090)** — bitsandbytes 4-bit kernels are still flaky there.

---

## Optional: Deploy to Ollama

```bash
ollama create trading-qwen-ft \
  -f backtest/data/models/qlora_cloud/Modelfile
```

`Modelfile` (auto-generated by train_qlora.py, or create manually):
```
FROM ./backtest/data/models/qlora_cloud/gguf/unsloth.Q4_K_M.gguf
PARAMETER temperature 0.1
SYSTEM "You are a Senior Technical Analyst..."
```

Then set `LLM_MODEL=trading-qwen-ft` in `.env` — the LangGraph agent picks it up via `llm_factory.py`.

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| OOM during training | Set `BATCH=4` or `BATCH=2` env var before `run_cloud.sh` |
| OOM during eval | Add `--eval-batch-size 1` |
| `FA2 = False` in banner | Still works via xformers, just ~20% slower |
| `ModuleNotFoundError: unsloth` | Use `python3`, not a custom venv python |
| DVC pull fails | Check `aws sts get-caller-identity` — credentials expired? |
| Zombie GPU process | `nvidia-smi` → note the PID → `kill -9 <PID>` |
| Process died mid-run | `bash run_cloud.sh --resume` — picks up from last checkpoint |
| "No space left on device" | `df -h /` — container disk full; clean pip cache: `pip cache purge` |
| torch version changed | `pip install 'torch==2.10.0' --index-url https://download.pytorch.org/whl/cu128` |
| Loss not decreasing | Verify dataset loaded (56161 lines), check LR is 2e-5 |
