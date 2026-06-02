# QLoRA Fine-Tuning on SageMaker

Step-by-step guide to fine-tune Qwen 2.5 7B using a SageMaker Notebook Instance.

## Cost Estimate

| Instance | GPU | VRAM | Cost/hr | ~2.5 hrs training |
|----------|-----|------|---------|-------------------|
| ml.g5.xlarge | A10G | 24GB | ~$1.41 | ~$3.50 |
| ml.g5.2xlarge | A10G | 24GB | ~$1.90 | ~$4.75 |

**Total cost: ~$3-5 USD** for the full training run.

---

## Step 1: Create a Notebook Instance

1. Go to **AWS Console → SageMaker → Notebook Instances → Create**
2. Configure:
   - **Name**: `qlora-trading`
   - **Instance type**: `ml.g5.xlarge` (24GB A10G GPU)
   - **Volume size**: 50 GB (model weights need space)
   - **IAM Role**: Create a new role or use an existing SageMaker role
3. Click **Create notebook instance**
4. Wait for status to become **InService** (~3-5 min)

---

## Step 2: Open Terminal

1. Click **Open JupyterLab**
2. Go to **File → New → Terminal**

---

## Step 3: Clone and Setup

```bash
# Clone repo
git clone https://github.com/alejandrogonzalz/trading_management.git
cd trading_management/langgraph

# Create venv with system Python (3.10+ already available on SageMaker)
python3 -m venv .venv --system-site-packages
source .venv/bin/activate

# Install training dependencies (see requirements-finetuning.txt for pins)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install datasets scikit-learn pyyaml

# Verify GPU is available
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
```

> **Note on candle data**: the evaluation step simulates trades against future
> candles (`backtest/data/candles/*.json`). If those files are not in the repo
> (they are DVC-tracked / gitignored), upload them manually or skip trade
> simulation with `--max-eval 0`. Direction accuracy is computed regardless.

---

## Step 4: Run Training

```bash
# Default config (recommended first run)
python optimization/train_qlora.py

# Or with specific hyperparameters
python optimization/train_qlora.py --lr 0.00002 --rank 16 --alpha 32 --epochs 3

# Or using the YAML config
python optimization/train_qlora.py --config optimization/configs/qlora.yaml

# Quick test (evaluate on 100 samples only)
python optimization/train_qlora.py --max-eval 100 --epochs 1
```

### What the script does:
1. **prepare_data** — splits `dataset.jsonl` with the same 70/15/15 temporal split
   used by LSTM/XGBoost (no sort) → `training_data/{train,val,test}.jsonl` in
   system/user/assistant chat format
2. **load_model** — loads Qwen 2.5 7B in 4-bit quantization (~4-5 GB VRAM) and
   applies LoRA adapters (rank=16, ~0.6% trainable params) on all attention +
   MLP projection layers
3. **train** — SFTTrainer: 3 epochs, effective batch=16, cosine LR, eval every
   100 steps on val set, saves best checkpoint by eval_loss
4. **save_model** — writes LoRA adapters + merged GGUF (Q4_K_M) for Ollama
5. **evaluate** — rebuilds prompts from raw indicators (NOT from the training
   JSONL), generates predictions with greedy decoding, runs `simulate_trade()`
   against future candles, emits `sample_keys` for paired McNemar / t-test
6. **save result** — writes `optimization/results/qlora_optimization.json` with
   full metrics (accuracy, win_rate, profit_factor, Sharpe, drawdown) +
   per-sample predictions/actuals/trade_results/sample_keys

### Expected output:
```
Training completed in ~90-150 min
Test accuracy: 0.XXXX
Result saved: optimization/results/qlora_optimization.json
Model saved: backtest/data/models/qlora_qwen25_7b/
```

---

## Step 5: Push Results

```bash
# Add results (NOT the model weights — too large for git)
git add optimization/results/qlora_optimization.json
git commit -m "feat(optimization): QLoRA fine-tuning results — Qwen 2.5 7B"
git push origin dev
```

---

## Step 6: STOP THE INSTANCE

**IMPORTANT**: You pay while the instance is running ($1.41/hr = $34/day).

1. Go to **SageMaker → Notebook Instances**
2. Select `qlora-trading`
3. Click **Stop**

---

## Optional: Deploy to Ollama

If the fine-tuned model outperforms zero-shot, you can load the GGUF into Ollama:

```bash
# On your production machine (with Ollama installed)
# Copy the GGUF file from SageMaker first
ollama create trading-qwen -f Modelfile
```

Where `Modelfile` contains:
```
FROM ./qlora_qwen25_7b/gguf/unsloth.Q4_K_M.gguf
PARAMETER temperature 0.1
SYSTEM "You are a Senior Technical Analyst..."
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| CUDA OOM | Reduce `--batch-size` to 2 or increase `gradient_accumulation_steps` |
| Slow training | Check GPU util with `nvidia-smi` — if low, increase batch size |
| Permission denied on git push | Configure git credentials: `git config credential.helper store` |
| Model downloads slowly | First run downloads ~4GB of weights; subsequent runs use cache |
| Instance won't start | Check service quota for ml.g5.xlarge in your region |

---

## Running Multiple Configs

To try all 5 recommended configs from `qlora.yaml`:

```bash
# Config 1 (default — recommended)
python optimization/train_qlora.py --lr 0.00002 --rank 16 --alpha 32 --epochs 2

# Config 2
python optimization/train_qlora.py --lr 0.00005 --rank 8 --alpha 16 --epochs 3 --batch-size 8

# Config 3
python optimization/train_qlora.py --lr 0.00001 --rank 32 --alpha 64 --epochs 2

# Config 4
python optimization/train_qlora.py --lr 0.0001 --rank 16 --alpha 32 --epochs 1 --batch-size 8

# Config 5
python optimization/train_qlora.py --lr 0.00005 --rank 16 --alpha 32 --epochs 2
```

Each run overwrites `qlora_optimization.json`. To keep all results, rename between runs:
```bash
cp optimization/results/qlora_optimization.json optimization/results/qlora_config1.json
```
