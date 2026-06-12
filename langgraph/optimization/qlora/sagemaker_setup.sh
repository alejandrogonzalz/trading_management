#!/bin/bash
# SageMaker environment setup — run once after opening the terminal.
#
# Usage (from JupyterLab terminal):
#   cd ~
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

# 1. Verify GPU
echo "[1/6] Checking GPU..."
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: No GPU detected. Did you select a GPU instance (ml.g5/g6e)?"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo ""

# 2. Check Python version
echo "[2/6] Checking Python..."
PYTHON=$(command -v python3.11 || command -v python3.10 || command -v python3)
if [ -z "$PYTHON" ]; then
    echo "ERROR: No Python 3.10+ found."
    exit 1
fi
echo "Using: $PYTHON ($($PYTHON --version))"
echo ""

# 3. Create venv
echo "[3/6] Creating virtual environment..."
if [ -d ".venv" ]; then
    echo "  .venv already exists — skipping creation."
else
    $PYTHON -m venv .venv --system-site-packages
    echo "  Created .venv"
fi
source .venv/bin/activate
echo "  Activated: $(which python)"
echo ""

# 4. Install dependencies
echo "[4/6] Installing training dependencies..."
pip install --upgrade pip -q

# Unsloth (pulls FA2 + triton on Linux automatically)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" -q

# TRL + friends (no-deps to avoid version conflicts with Unsloth)
pip install --no-deps trl peft accelerate bitsandbytes -q

# Additional deps the script needs
pip install datasets scikit-learn pyyaml tqdm matplotlib httpx -q

echo "  Done."
echo ""

# 5. Verify imports + FA2
echo "[5/6] Verifying installation..."
python -c "
import torch
import sys

gpu_name = torch.cuda.get_device_name(0)
vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
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

# Check training deps
from trl import SFTTrainer
from datasets import Dataset
print(f'  TRL + datasets: OK ✓')
print()
print(f'  Python: {sys.version}')
"
echo ""

# 6. Create logs directory
echo "[6/6] Creating directories..."
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
