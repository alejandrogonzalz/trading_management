#!/bin/bash
# RunPod GPU setup for QLoRA fine-tuning. Handles CUDA driver/torch version
# mismatch that occurs when the host driver is older than what Unsloth's default
# torch build expects (e.g. driver 550.x → CUDA 12.4, but torch ships cu128).
#
# Tested on: RunPod A100 80GB PCIe (driver 550.127.05 → CUDA 12.4 max)
#
# Usage (inside RunPod terminal):
#   cd /trading_management/langgraph   # or wherever you cloned
#   bash optimization/qlora/setup_runpod.sh
#
# After setup:
#   source .venv/bin/activate
#   aws configure                       # needed for DVC S3
#   dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles
#   python optimization/qlora/train_qlora.py --max-steps 3 --max-eval 20 --eval-batch-size 8 --diagnostic-samples 0
#   # then: nohup bash optimization/qlora/run_cloud.sh > optimization/qlora/logs/run_cloud.log 2>&1 &

set -e

echo "============================================================"
echo "  RunPod QLoRA Setup"
echo "  $(date)"
echo "============================================================"
echo ""

# 0. PATH fixups for CUDA tools
for p in /usr/local/cuda/bin /opt/pytorch/bin /usr/local/bin; do
    [ -d "$p" ] && export PATH="$p:$PATH"
done
if ! grep -q '/usr/local/cuda/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
fi

# 1. GPU + driver detection
echo "[1/7] Checking GPU + NVIDIA driver..."
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is this a GPU pod?"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# Extract max CUDA version supported by the driver
DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
echo "  Host driver supports up to CUDA: $DRIVER_CUDA"
echo ""

# Determine the right torch CUDA index URL
# torch builds: cu118, cu121, cu124, cu126 (torch 2.7+), cu128 (torch 2.8+)
CUDA_MAJOR=$(echo "$DRIVER_CUDA" | cut -d. -f1)
CUDA_MINOR=$(echo "$DRIVER_CUDA" | cut -d. -f2)

if [ "$CUDA_MAJOR" -gt 12 ] || ([ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 8 ]); then
    TORCH_CUDA="cu128"
    TORCH_VER="2.8.0"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 6 ]; then
    TORCH_CUDA="cu126"
    TORCH_VER="2.7.0"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
    TORCH_CUDA="cu124"
    TORCH_VER="2.6.0"
elif [ "$CUDA_MAJOR" -eq 12 ] && [ "$CUDA_MINOR" -ge 1 ]; then
    TORCH_CUDA="cu121"
    TORCH_VER="2.6.0"
else
    echo "ERROR: CUDA $DRIVER_CUDA is too old (need >=12.1). Redeploy on a newer host."
    exit 1
fi
echo "  → Will install torch ${TORCH_VER}+${TORCH_CUDA}"
echo ""

# 2. Python
echo "[2/7] Checking Python + dev headers..."
PYTHON=""
for candidate in python3.12 python3.11 python3.10; do
    command -v "$candidate" &>/dev/null && PYTHON="$candidate" && break
done
[ -z "$PYTHON" ] && PYTHON=$(command -v python3)
echo "  Using: $PYTHON ($($PYTHON --version))"

PYVER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq 2>/dev/null || true
    sudo apt-get install -y -q "python${PYVER}-dev" 2>/dev/null || \
        sudo apt-get install -y -q python3-dev 2>/dev/null || \
        echo "  WARNING: could not install python-dev — Triton may fail at runtime"
fi
echo ""

# 3. Clean venv
echo "[3/7] Creating virtual environment..."
[ -d .venv ] && { echo "  Removing existing .venv..."; rm -rf .venv; }
$PYTHON -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
echo "  Activated: $(which python)"
echo ""

# 4. Install torch FIRST (pinned to the driver-compatible CUDA version)
echo "[4/7] Installing PyTorch ${TORCH_VER}+${TORCH_CUDA} (matched to host driver)..."
pip install torch==${TORCH_VER} torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
echo ""

# 5. Install Unsloth + training deps
echo "[5/7] Installing Unsloth + training dependencies..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Re-pin torch in case Unsloth pulled a different CUDA build
CURRENT_CUDA=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || echo "none")
if [ "$CURRENT_CUDA" != "12.4" ] && [ "$CURRENT_CUDA" != "none" ]; then
    echo "  Unsloth overrode torch (now CUDA $CURRENT_CUDA). Re-pinning to ${TORCH_CUDA}..."
    pip install --force-reinstall torch==${TORCH_VER} torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
    pip install -U bitsandbytes
fi

# Remove torchao — it's incompatible with torch <2.7 (uses register_constant API)
# and we don't need it (bitsandbytes handles our 4-bit quantization, not torchao).
echo "  Removing torchao (incompatible with torch 2.6, not needed for QLoRA)..."
pip uninstall torchao -y 2>/dev/null || true

# FlashAttention-2 (optional — try to install, non-fatal)
echo ""
echo "  Attempting FlashAttention-2 install (optional, may fail)..."
if pip install flash-attn --no-build-isolation 2>/dev/null; then
    echo "  FlashAttention-2: INSTALLED"
else
    echo "  FlashAttention-2: SKIPPED (Unsloth will use Triton kernels — slightly slower)"
fi

pip install "dvc[s3]" scikit-learn pyyaml tqdm matplotlib httpx -q
echo "  Done."
echo ""

# 6. AWS CLI for DVC
echo "[6/7] Checking AWS CLI + tmux..."
export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
hash -r
if ! command -v aws &>/dev/null; then
    echo "  Installing AWS CLI v2 (standalone binary)..."
    curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -qo /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install --update 2>/dev/null || \
        /tmp/aws/install --install-dir "$HOME/.local/aws-cli" --bin-dir "$HOME/.local/bin" --update
    rm -rf /tmp/awscliv2.zip /tmp/aws
    hash -r
fi
echo "  aws: $(aws --version)"
if aws sts get-caller-identity &>/dev/null; then
    echo "  AWS OK: $(aws sts get-caller-identity --query 'Arn' --output text)"
else
    echo "  ⚠ No AWS creds yet. Run 'aws configure' before 'dvc pull'."
fi

if command -v tmux &>/dev/null; then
    echo "  tmux: $(tmux -V)"
else
    sudo apt-get install -y tmux -qq 2>/dev/null && echo "  tmux: installed" || \
        echo "  tmux: not available — use nohup"
fi
echo ""

# 7. Verify everything
echo "[7/7] Verifying installation..."
python -c "
import torch
print(f'  PyTorch:  {torch.__version__}')
print(f'  CUDA:     {torch.version.cuda}')
print(f'  GPU:      {torch.cuda.get_device_name(0)}  (cc={torch.cuda.get_device_capability(0)})')
print(f'  GPU OK:   {torch.cuda.is_available()}')
try:
    import flash_attn
    print(f'  FA2:      v{flash_attn.__version__}')
except ImportError:
    print('  FA2:      not installed (Triton fallback)')
from unsloth import FastLanguageModel
print('  Unsloth:  OK')
import trl, transformers, peft
print(f'  TRL:      {trl.__version__}')
print(f'  HF:       {transformers.__version__}')
print(f'  PEFT:     {peft.__version__}')
"
echo ""

mkdir -p optimization/qlora/logs optimization/qlora/results

echo "============================================================"
echo "  Setup complete! Next steps:"
echo ""
echo "    source .venv/bin/activate"
echo "    aws configure                  # Access Key + Secret for DVC S3"
echo "    dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles"
echo ""
echo "    # Smoke test (3 steps, fast):"
echo "    python optimization/qlora/train_qlora.py --max-steps 3 --max-eval 20 --eval-batch-size 8 --diagnostic-samples 0"
echo ""
echo "    # Full training:"
echo "    nohup bash optimization/qlora/run_cloud.sh > optimization/qlora/logs/run_cloud.log 2>&1 &"
echo "    echo \"PID: \$!\"   # tail -f optimization/qlora/logs/run_cloud.log"
echo "============================================================"
