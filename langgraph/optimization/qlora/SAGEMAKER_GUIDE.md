# QLoRA Fine-Tuning on SageMaker

Step-by-step guide to fine-tune Qwen 2.5 7B using a SageMaker Notebook Instance.

## Cost Estimate

| Instance | GPU | VRAM | Cost/hr | 3 epochs (~3-6h) |
|----------|-----|------|---------|-------------------|
| **ml.g6e.xlarge** | L40S | 48GB | ~$2.00 | ~$6-12 |
| ml.g5.xlarge | A10G | 24GB | ~$1.41 | ~$4-8 |

**Recommended: `ml.g6e.xlarge`** — 48GB lets you run `batch_size=8` comfortably at
`max_seq_length=1024`, and FlashAttention-2 installs cleanly on Linux (the real
speedup). The g5.xlarge works but is tighter — use `batch_size=4` there.

---

## Step 1: Create a Notebook Instance

1. Go to **AWS Console → SageMaker → Notebook Instances → Create**
2. Configure:
   - **Name**: `qlora-trading`
   - **Instance type**: `ml.g6e.xlarge` (48GB L40S GPU)
   - **Volume size**: 50 GB (model weights + checkpoints need space)
   - **IAM Role**: Create a new role or use an existing SageMaker role
   - **Lifecycle config** (optional but recommended): set an idle-shutdown script
     so the instance auto-stops after 60 min of inactivity
3. Click **Create notebook instance**
4. Wait for status to become **InService** (~3-5 min)

---

## Step 2: Open Terminal

1. Click **Open JupyterLab**
2. Go to **File → New → Terminal**

---

## Step 3: Clone and Setup

**IMPORTANT**: Clone into `~/SageMaker/` — it's the only persistent volume. Everything
else is wiped on stop/start.

```bash
cd ~/SageMaker
git clone https://github.com/alejandrogonzalz/trading_management.git
cd trading_management/langgraph

# Run the automated setup script (installs everything + pulls data + verifies)
bash optimization/qlora/sagemaker_setup.sh
```

The setup script handles:
- GPU + Python version verification
- venv creation with pinned TRL/transformers versions (API-compatible)
- DVC install + `dvc pull` for the dataset and candles
- HuggingFace cache in the persistent volume
- FA2 + Unsloth + SFTTrainer API compatibility check

> **Confirm FA2**: when the model loads, the startup banner should show `FA2 = True`.
> That's the main speedup vs local Windows (where FA2 won't install).

> **Dataset**: `dataset.jsonl` is DVC-tracked. The setup script pulls it automatically.
> If DVC pull fails (IAM permissions), upload the file manually via JupyterLab to
> `backtest/data/labeled/dataset.jsonl`.

---

## Step 4: Smoke Test (~2 min, ~$0.05)

Before committing to a multi-hour run, confirm the environment works:

```bash
python optimization/qlora/train_qlora.py --max-steps 3 --max-eval 10
```

This loads the model, runs 3 training steps, evaluates 10 samples, and exits.
If it passes, the full run will too — the failure mode is always step 0.

---

## Step 5: Run Training

Use **tmux** so the job survives browser disconnects:

```bash
source .venv/bin/activate

# REQUIRED — pull data first (no candles → win_rate/PF/Sharpe = 0):
dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles

# Single cloud run on the fixed split (recommended), survives disconnects via nohup:
nohup bash optimization/qlora/run_cloud.sh > optimization/qlora/logs/run_cloud.log 2>&1 &
echo "PID: $!"

# Follow progress:
tail -f optimization/qlora/logs/run_cloud.log
```

### What happens:
1. **prepare_data** — splits `dataset.jsonl` with the same 70/15/15 temporal split
   used by LSTM/XGBoost (no sort) → `training_data/{train,val,test}.jsonl`
2. **load_model** — loads Qwen 2.5 7B in 4-bit (~4-5 GB VRAM) + LoRA adapters
3. **train** — SFTTrainer: 3 epochs, effective batch=128 (8×16), cosine LR, eval
   every 250 steps on val set, saves best checkpoint by eval_loss
4. **save_model** — LoRA adapters + merged GGUF (Q4_K_M) for Ollama
5. **evaluate** — greedy decoding on all 8,425 test samples, trade simulation,
   emits `sample_keys` for paired McNemar / t-test
6. **save result** → `optimization/qlora/results/qlora_optimization.json`

### If the run is interrupted:

```bash
# Resume from the last checkpoint (saves every 250 steps)
python optimization/qlora/train_qlora.py --epochs 3 --batch-size 8 --resume 2>&1 | tee -a optimization/qlora/logs/qlora_sagemaker.log
```

`--resume` finds the latest `checkpoint-N` in the output directory and continues
from there. No work is lost.

### Expected timing (ml.g6e.xlarge):
- ~2-4 s/step (vs ~28 s/step local)
- ~1-2 h per epoch
- **~3-6 h total** for 3 epochs + evaluation

---

## Step 6: Push Results

```bash
# Add results (NOT the model weights — too large for git)
git add optimization/qlora/results/qlora_optimization.json
git commit -m "feat(qlora): 3-epoch fine-tuning results — Qwen 2.5 7B on SageMaker"
git push origin dev
```

---

## Step 7: STOP THE INSTANCE

**IMPORTANT**: You pay while the instance is running (~$2/hr = $48/day).

1. Go to **SageMaker → Notebook Instances**
2. Select `qlora-trading`
3. Click **Stop**

If you set up the idle-shutdown lifecycle config in Step 1, it auto-stops after
60 min of no terminal/notebook activity — but don't rely on it alone.

---

## Step 8: Compare Results Locally

Back on your machine:

```bash
cd langgraph
git pull origin dev

# Paired statistical test vs LSTM (the current winner at 81.5%)
python -m cli compare-stats \
  --a optimization/qlora/results/qlora_optimization.json \
  --b backtest/data/results/ml-lstm.json
```

---

## Optional: Deploy to Ollama

If the fine-tuned model outperforms zero-shot, load the GGUF into Ollama:

```bash
# Copy the GGUF file from SageMaker first (scp or S3)
ollama create trading-qwen-ft -f Modelfile
```

Where `Modelfile` contains:
```
FROM ./qlora_qwen25_7b/gguf/unsloth.Q4_K_M.gguf
PARAMETER temperature 0.1
SYSTEM "You are a Senior Technical Analyst..."
```

Then set `LLM_MODEL=trading-qwen-ft` in your environment — the agent uses it
automatically via `llm_factory.py`.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM | Reduce `--batch-size` to 4 (g5) or 2, increase `gradient_accumulation_steps` proportionally |
| Slow training (low GPU util) | Check `nvidia-smi` — if GPU util < 80%, increase batch size |
| FA2 = False in banner | Reinstall: `pip install flash-attn --no-build-isolation` |
| Kernel/terminal dies | Use tmux (Step 5). Reattach with `tmux attach -t qlora` |
| Interrupted mid-training | Re-run with `--resume` — picks up from last checkpoint |
| Permission denied on git push | `git config credential.helper store` then push again |
| Model downloads slowly | First run downloads ~4GB; cached afterwards |
| Instance won't start | Check service quota for ml.g6e.xlarge in your region (request increase if needed) |
| Checkpoint disk full | `save_total_limit=3` — only 3 checkpoints kept. If 50GB runs low, increase volume |

---

## Single model, not a sweep

The thesis trains **one** defensible cloud model on the fixed strict-temporal
split — the old multi-config sweep was removed (its results are archived under
`results/archive/` and must not be cited). Use the wrapper:

```bash
bash optimization/qlora/run_cloud.sh
```

Or invoke the script directly (equivalent to the wrapper):
```bash
# Recommended config on the fixed split (full test eval, no --max-eval)
python optimization/qlora/train_qlora.py --lr 0.00002 --rank 16 --alpha 32 --epochs 3 --batch-size 2 --grad-accum 8 --tag qlora_cloud
```

Each run writes `results/qlora_<tag>.json` (per-run) plus a canonical
`results/qlora_optimization.json` that `compare-stats` reads by default. Use a
distinct `--tag` (and `--output-dir`) if you ever do try a second config so the
per-run files don't collide.
