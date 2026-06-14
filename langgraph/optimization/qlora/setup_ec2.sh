#!/bin/bash
# Single GPU-instance setup for QLoRA fine-tuning — run once. Targets EC2 + the
# Deep Learning AMI (where FlashAttention-2 builds), but is self-contained enough
# to also work on a SageMaker Studio terminal (FA2 install is non-fatal there).
#
# Recommended instance:
#   AMI:      "Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 24.04)"
#             (ships a MATCHED NVIDIA driver + CUDA toolkit → FlashAttention-2 builds)
#   Type:     g6e.xlarge  (NVIDIA L40S 48GB, Ada Lovelace sm_89, ~$1.86/hr on-demand)
#   Storage:  100 GB gp3
#   SG:       SSH (22) from your IP only
#
# Usage:
#   ssh -i ~/.ssh/your-key.pem ubuntu@<public-ip>
#   git clone https://github.com/alejandrogonzalz/trading_management.git
#   cd trading_management/langgraph
#   bash optimization/qlora/setup_ec2.sh
#
# After setup, train with nohup (works whether or not tmux is present):
#   source .venv/bin/activate
#   dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles
#   nohup bash optimization/qlora/run_cloud.sh > optimization/qlora/logs/run_cloud.log 2>&1 &
#   echo "PID: $!"

set -e

echo "============================================================"
echo "  EC2 QLoRA Setup (Deep Learning AMI)"
echo "  $(date)"
echo "============================================================"
echo ""

# 0. Ensure nvidia-smi and CUDA tools are in PATH. The DLAMI puts them under
#    /opt/pytorch/bin or /usr/local/cuda/bin — not always in the default PATH.
for p in /opt/pytorch/bin /usr/local/cuda/bin /usr/local/bin; do
    [ -d "$p" ] && export PATH="$p:$PATH"
done
# Persist for future SSH sessions (idempotent — only adds if not already there).
if ! grep -q '/opt/pytorch/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH="/opt/pytorch/bin:/usr/local/cuda/bin:$PATH"' >> ~/.bashrc
fi

# 1. Verify GPU + driver
echo "[1/7] Checking GPU + NVIDIA driver..."
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Did you launch a g6e/g5 instance with the Deep Learning AMI?"
    echo "  Tried: /opt/pytorch/bin, /usr/local/cuda/bin, /usr/local/bin"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# 2. Python 3.10+ (the AMI ships 3.11/3.12)
echo "[2/7] Checking Python..."
PYTHON=""
for candidate in python3.12 python3.11 python3.10; do
    command -v "$candidate" &>/dev/null && PYTHON="$candidate" && break
done
[ -z "$PYTHON" ] && PYTHON=$(command -v python3)
echo "Using: $PYTHON ($($PYTHON --version))"
echo ""

# 3. Clean venv (no --system-site-packages — keep the base env's libs from leaking in)
echo "[3/7] Creating virtual environment..."
[ -d .venv ] && { echo "  Removing existing .venv..."; rm -rf .venv; }
$PYTHON -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q
echo "  Activated: $(which python)"
echo ""

# 4. Dependencies. Unsloth (from git) resolves its own torch/TRL/transformers/peft.
echo "[4/7] Installing training dependencies..."
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# FA2 requires: (a) CUDA_HOME with nvcc, (b) Python dev headers, (c) ninja for
# fast parallel compilation. The DLAMI runtime-only images often ship the driver
# but not nvcc or Python.h; detect and install what's missing.
echo "  Installing build prerequisites for FlashAttention-2..."
sudo apt-get update -qq
# Python dev headers (Python.h) — required for the C++ extension build
sudo apt-get install -y python3-dev python3.12-dev -qq 2>/dev/null || true
# ninja — parallel build (~3 min instead of ~15 min without it)
pip install ninja -q

# CUDA toolkit (nvcc) — needed for compiling CUDA kernels
NVCC_PATH=$(find /usr/local -name "nvcc" -type f 2>/dev/null | head -1)
if [ -z "$NVCC_PATH" ]; then
    echo "  nvcc not found — installing CUDA toolkit (this takes ~2 min)..."
    sudo apt-get install -y cuda-toolkit -qq 2>/dev/null || \
        sudo apt-get install -y nvidia-cuda-toolkit -qq 2>/dev/null || true
    NVCC_PATH=$(find /usr/local -name "nvcc" -type f 2>/dev/null | head -1)
fi

if [ -n "$NVCC_PATH" ]; then
    export CUDA_HOME=$(dirname "$(dirname "$NVCC_PATH")")
    echo "  CUDA_HOME=$CUDA_HOME (nvcc: $NVCC_PATH)"
    # MAX_JOBS=2 limits parallel compilation to 2 cores — leaves headroom for SSH
    # on small instances (4 vCPUs). Without this, FA2 compilation saturates all
    # cores and freezes the machine for ~15 min (can't even Ctrl+C or SSH in).
    MAX_JOBS=2 pip install flash-attn --no-build-isolation || \
        echo "  WARNING: flash-attn build failed — training will run WITHOUT FA2 (slower)."
    # Persist CUDA_HOME for future sessions
    grep -q "CUDA_HOME" ~/.bashrc 2>/dev/null || \
        echo "export CUDA_HOME=$CUDA_HOME" >> ~/.bashrc
else
    echo "  WARNING: nvcc not found even after toolkit install — FA2 skipped (training will be ~2-3x slower)."
fi

pip install "dvc[s3]" scikit-learn pyyaml tqdm matplotlib httpx -q
echo "  Done."
echo ""

# 5. AWS creds for DVC S3 (EC2: prefer an attached IAM role; else aws configure)
echo "[5/7] Checking AWS access for DVC..."
if aws sts get-caller-identity &>/dev/null; then
    echo "  AWS OK: $(aws sts get-caller-identity --query 'Arn' --output text)"
else
    echo "  No AWS creds. Attach an IAM role with s3:GetObject on s3://trading-management-dvc/,"
    echo "  or run 'aws configure'."
fi
echo ""

# 6. Install tmux (EC2 has apt, unlike SageMaker — survives SSH drops better than nohup)
echo "[6/7] Installing tmux..."
if command -v tmux &>/dev/null; then
    echo "  tmux already installed: $(tmux -V)"
else
    sudo apt-get install -y tmux -qq 2>/dev/null && echo "  Installed: $(tmux -V)" || \
        echo "  WARNING: tmux install failed — use nohup instead"
fi
echo ""

# 7. Verify torch + CUDA + FA2 + Unsloth
echo "[7/7] Verifying installation..."
python -c "
import torch
print(f'  GPU: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}')
print(f'  CUDA (torch): {torch.version.cuda}   PyTorch: {torch.__version__}')
try:
    import flash_attn; print(f'  FlashAttention-2: v{flash_attn.__version__}  (expect FA2=True at train time)')
except ImportError:
    print('  FlashAttention-2: NOT INSTALLED — training will be ~2-3x slower')
from unsloth import FastLanguageModel; print('  Unsloth: OK')
import trl, transformers, peft
print(f'  TRL {trl.__version__}  transformers {transformers.__version__}  peft {peft.__version__}')
"
echo ""

mkdir -p optimization/qlora/logs optimization/qlora/results

echo "============================================================"
echo "  Setup complete. Next:"
echo ""
echo "    source .venv/bin/activate"
echo "    dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles"
echo "    python optimization/qlora/train_qlora.py --max-steps 3 --max-eval 10   # smoke test"
echo ""
echo "  Then train (pick one):"
echo "    # Option A: tmux (recommended — reattach after SSH drop)"
echo "    tmux new -s qlora"
echo "    bash optimization/qlora/run_cloud.sh 2>&1 | tee optimization/qlora/logs/run_cloud.log"
echo "    # Ctrl+B, D to detach; tmux attach -t qlora to reattach"
echo ""
echo "    # Option B: nohup (simpler, no reattach)"
echo "    nohup bash optimization/qlora/run_cloud.sh > optimization/qlora/logs/run_cloud.log 2>&1 &"
echo "    echo \"PID: \$!\"   # tail -f optimization/qlora/logs/run_cloud.log"
echo ""
echo "  STOP the instance when done (keeps disk, no GPU charge):"
echo "    aws ec2 stop-instances --instance-ids i-xxxx --region us-east-1"
echo "============================================================"
