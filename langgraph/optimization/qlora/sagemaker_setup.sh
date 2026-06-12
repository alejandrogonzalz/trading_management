#!/bin/bash
# SageMaker environment setup — run once after opening the terminal.
#
# IMPORTANT: Clone into ~/SageMaker/ (the only persistent volume on Notebook Instances).
# Everything outside ~/SageMaker/ is lost on stop/start.
#
# Usage (from JupyterLab terminal):
#   cd ~/SageMaker
#   git clone https://github.com/alejandrogonzalz/trading_management.git
#   cd trading_management/langgraph
#   bash optimization/qlora/sagemaker_setup.sh
#
# After this completes, run the training:
#   tmux new -s qlora
#   source .venv/bin/activate
#   bash optimization/qlora/run_qlora_search.sh

set -e

echo "============================================================"
echo "  SageMaker QLoRA Setup"
echo "  $(date)"
echo "============================================================"
echo ""

# 0. Verify we're in the persistent volume
if [[ "$PWD" != */SageMaker/* && "$PWD" != /home/ec2-user/SageMaker* ]]; then
    echo "WARNING: You are NOT inside ~/SageMaker/."
    echo "  Files outside ~/SageMaker/ are LOST on instance stop/start."
    echo "  Recommended: cd ~/SageMaker && git clone ... && cd trading_management/langgraph"
    echo ""
    read -p "  Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 1. Verify GPU
echo "[1/8] Checking GPU..."
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU detected. Did you select a GPU instance (ml.g5/g6e)?"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 2. Check Python version (require 3.10+ for PEP 604 syntax)
echo "[2/8] Checking Python..."
PYTHON=""
for candidate in python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    # Fallback: check if python3 is >= 3.10
    PY3=$(command -v python3)
    if [ -n "$PY3" ]; then
        PY_VER=$($PY3 -c "import sys; print(f'{sys.version_info.minor}')")
        if [ "$PY_VER" -ge 10 ] 2>/dev/null; then
            PYTHON="$PY3"
        fi
    fi
fi
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ required (for PEP 604 type syntax). Found:"
    python3 --version 2>/dev/null || echo "  no python3"
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version))"
echo ""

# 3. Create venv
echo "[3/8] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "  .venv already exists — skipping creation."
else
    $PYTHON -m venv .venv --system-site-packages
    echo "  Created .venv"
fi
source .venv/bin/activate
echo "  Activated: $(which python)"
echo ""

# 4. Set HuggingFace cache to persistent volume (survives stop/start)
export HF_HOME="$HOME/SageMaker/.cache/huggingface"
mkdir -p "$HF_HOME"
echo "[4/8] HF cache: $HF_HOME"
# Persist this across sessions
grep -q "HF_HOME" ~/.bashrc 2>/dev/null || echo "export HF_HOME=$HF_HOME" >> ~/.bashrc
echo ""

# 5. Install dependencies (pinned versions from requirements-finetuning.txt)
echo "[5/8] Installing training dependencies..."
pip install --upgrade pip -q

# Unsloth (pulls FA2 + triton on Linux automatically)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q

# TRL + friends — pinned to versions compatible with train_qlora.py's API
# (SFTTrainer(tokenizer=, dataset_text_field=, max_seq_length=, packing=))
pip install "trl>=0.8.6,<0.10" "peft>=0.11.0" "accelerate>=0.30.0" "bitsandbytes>=0.43.0" -q
pip install "transformers>=4.43.0,<4.46" -q

# Additional deps the script needs
pip install "dvc[s3]" datasets scikit-learn pyyaml tqdm matplotlib httpx -q

echo "  Done."
echo ""

# 6. Configure AWS credentials (required for DVC S3 remote)
echo "[6/9] Configuring AWS credentials for DVC S3 access..."
if aws sts get-caller-identity &>/dev/null; then
    echo "  AWS credentials already configured ✓"
    aws sts get-caller-identity --query 'Arn' --output text
else
    echo "  AWS credentials needed to pull data from s3://trading-management-dvc/"
    echo "  Running 'aws configure' — enter your Access Key, Secret Key, region (us-east-1):"
    echo ""
    aws configure
    echo ""
    if aws sts get-caller-identity &>/dev/null; then
        echo "  AWS credentials configured ✓"
    else
        echo "  WARNING: AWS auth failed. DVC pull will fail."
        echo "  You can re-run 'aws configure' manually later."
    fi
fi
echo ""

# 7. Pull dataset via DVC (requires S3 access)
echo "[7/9] Pulling dataset via DVC..."
DATASET="backtest/data/labeled/dataset.jsonl"
if [ -f "$DATASET" ]; then
    LINES=$(wc -l < "$DATASET")
    echo "  Dataset already present: $LINES lines"
else
    if dvc pull "$DATASET.dvc" 2>/dev/null; then
        LINES=$(wc -l < "$DATASET")
        echo "  Pulled via DVC: $LINES lines"
    else
        echo "  WARNING: DVC pull failed. Possible causes:"
        echo "    - AWS credentials not configured (run 'aws configure')"
        echo "    - IAM user lacks s3:GetObject on s3://trading-management-dvc/"
        echo "    - .dvc/config remote not configured"
        echo ""
        echo "  Manual fix: upload dataset.jsonl via JupyterLab file browser to:"
        echo "    $PWD/$DATASET"
        echo ""
        echo "  Continuing setup — training will fail at prepare_data() without this file."
    fi
fi
# Also try pulling candles (needed for trade simulation in evaluate())
if [ -f "backtest/data/candles.dvc" ]; then
    dvc pull backtest/data/candles.dvc 2>/dev/null && echo "  Candles pulled." || \
        echo "  Candles DVC pull failed (eval will skip trade sim, accuracy still works)."
fi
echo ""

# 8. Verify imports + FA2
echo "[8/9] Verifying installation..."
python -c "
import torch
import sys

gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f'  GPU: {gpu_name} ({vram_gb:.1f} GB)')
print(f'  CUDA: {torch.version.cuda}')
print(f'  PyTorch: {torch.__version__}')

# Check FA2
try:
    import flash_attn
    print(f'  FlashAttention-2: v{flash_attn.__version__} ✓')
except ImportError:
    print('  FlashAttention-2: NOT INSTALLED')
    print('    → Try: pip install flash-attn --no-build-isolation')

# Check Unsloth
from unsloth import FastLanguageModel
print(f'  Unsloth: OK ✓')

# Check training deps — verify API compatibility
import trl, transformers, peft
print(f'  TRL: {trl.__version__}  transformers: {transformers.__version__}  peft: {peft.__version__}')
from trl import SFTTrainer
from datasets import Dataset
import inspect
sig = inspect.signature(SFTTrainer.__init__)
assert 'tokenizer' in sig.parameters or 'processing_class' in sig.parameters, \
    'SFTTrainer API mismatch — check TRL version'
if 'tokenizer' not in sig.parameters:
    print('  ⚠ TRL uses processing_class= not tokenizer= — version may be too new!')
    sys.exit(1)
print(f'  SFTTrainer API: OK ✓')
print()
print(f'  Python: {sys.version}')
"
echo ""

# 9. Create directories
echo "[9/9] Creating directories..."
mkdir -p optimization/qlora/logs optimization/qlora/results
echo "  optimization/qlora/{logs,results}/ ready."
echo ""

# Summary
echo "============================================================"
echo "  Setup complete! Next steps:"
echo ""
echo "  1. Start a tmux session:"
echo "       tmux new -s qlora"
echo ""
echo "  2. Activate the venv:"
echo "       source .venv/bin/activate"
echo ""
echo "  3. Smoke test (2 min):"
echo "       python optimization/qlora/train_qlora.py --max-steps 3 --max-eval 10"
echo ""
echo "  4. Run the full search (~15-30h):"
echo "       bash optimization/qlora/run_qlora_search.sh"
echo ""
echo "  5. When done: STOP THE INSTANCE in the AWS Console!"
echo "============================================================"
