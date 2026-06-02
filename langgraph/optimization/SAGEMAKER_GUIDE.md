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

# Install training dependencies
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes
pip install datasets scikit-learn pyyaml

# Verify GPU is available
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')"
```

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
1. Exports `dataset.jsonl` → `training_data/{train,val,test}.jsonl` (chat format)
2. Loads Qwen 2.5 7B with 4-bit quantization (~5GB VRAM)
3. Applies LoRA adapters (rank=16 → ~0.1% trainable params)
4. Trains with SFTTrainer (eval every 100 steps, saves best checkpoint)
5. Saves LoRA adapters + GGUF export for Ollama
6. Evaluates on test set (direction accuracy)
7. Writes result to `optimization/results/qlora_optimization.json`

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
