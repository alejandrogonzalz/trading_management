#!/bin/bash
# SageMaker Studio JupyterLab environment setup — run once after cloning.
#
# Usage (from JupyterLab terminal):
#   cd ~
#   git clone https://github.com/alejandrogonzalz/trading_management.git
#   cd trading_management/langgraph
#   bash optimization/qlora/sagemaker_setup.sh
#
# After this completes, run the training:
#   source .venv/bin/activate
#   dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles
#   nohup bash optimization/qlora/run_cloud.sh > optimization/qlora/logs/run_cloud.log 2>&1 &

set -e

echo "============================================================"
echo "  SageMaker QLoRA Setup"
echo "  $(date)"
echo "============================================================"
echo ""

# 1. Verify GPU
echo "[1/7] Checking GPU..."
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU detected. Did you select a GPU instance (ml.g5/g6e)?"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 2. Check Python version (require 3.10+ for PEP 604 syntax)
echo "[2/7] Checking Python..."
PYTHON=""
for candidate in python3.12 python3.11 python3.10; do
    if command -v "$candidate" &>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    PY3=$(command -v python3)
    if [ -n "$PY3" ]; then
        PY_VER=$($PY3 -c "import sys; print(sys.version_info.minor)")
        if [ "$PY_VER" -ge 10 ] 2>/dev/null; then
            PYTHON="$PY3"
        fi
    fi
fi
if [ -z "$PYTHON" ]; then
    echo "ERROR: Python 3.10+ required. Found:"
    python3 --version 2>/dev/null || echo "  no python3"
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version))"
echo ""

# 3. Create CLEAN venv (no --system-site-packages to avoid TF/Keras contamination)
echo "[3/7] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "  Removing existing .venv (may be corrupted)..."
    rm -rf .venv
fi
$PYTHON -m venv .venv
source .venv/bin/activate
echo "  Created and activated: $(which python)"
echo ""

# 4. Install dependencies
echo "[4/7] Installing training dependencies..."
pip install --upgrade pip -q

# Unsloth from git — let it resolve its own compatible TRL/transformers/peft versions.
# Do NOT pin TRL/transformers separately; Unsloth's setup.py handles the coupling.
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# FlashAttention-2 (the main speedup on Linux)
pip install flash-attn --no-build-isolation

# Additional deps the script needs (not pulled by Unsloth)
pip install "dvc[s3]" scikit-learn pyyaml tqdm matplotlib httpx -q

echo "  Done."
echo ""

# 5. Configure AWS credentials (required for DVC S3 remote)
echo "[5/7] Configuring AWS credentials for DVC S3 access..."
if aws sts get-caller-identity &>/dev/null; then
    echo "  AWS credentials already configured:"
    aws sts get-caller-identity --query 'Arn' --output text
else
    echo "  AWS credentials needed to pull data from s3://trading-management-dvc/"
    echo "  Running 'aws configure' — enter your Access Key, Secret Key, region (us-east-1):"
    echo ""
    aws configure
    echo ""
    if aws sts get-caller-identity &>/dev/null; then
        echo "  AWS credentials configured"
    else
        echo "  WARNING: AWS auth failed. DVC pull will fail."
        echo "  You can re-run 'aws configure' manually later."
    fi
fi
echo ""

# 6. Pull dataset via DVC (requires S3 access)
echo "[6/7] Pulling dataset via DVC..."
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
        echo "    - IAM role lacks s3:GetObject on s3://trading-management-dvc/"
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

# 7. Verify imports + FA2
echo "[7/7] Verifying installation..."
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
    print(f'  FlashAttention-2: v{flash_attn.__version__}')
except ImportError:
    print('  FlashAttention-2: NOT INSTALLED (training will be slower)')

# Check Unsloth + TRL
from unsloth import FastLanguageModel
print(f'  Unsloth: OK')

import trl, transformers, peft
print(f'  TRL: {trl.__version__}  transformers: {transformers.__version__}  peft: {peft.__version__}')

from trl import SFTTrainer
print(f'  SFTTrainer: OK')
print()
print(f'  Python: {sys.version}')
"
echo ""

# Create output directories
mkdir -p optimization/qlora/logs optimization/qlora/results

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
echo "  4. Pull data, then run the single cloud training (~3-6h):"
echo "       dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles"
echo "       nohup bash optimization/qlora/run_cloud.sh > optimization/qlora/logs/run_cloud.log 2>&1 &"
echo ""
echo "  5. When done: STOP THE INSTANCE in the AWS Console!"
echo "============================================================"
