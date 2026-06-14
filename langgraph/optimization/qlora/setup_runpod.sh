#!/bin/bash
# RunPod GPU setup for QLoRA fine-tuning.
#
# Handles the torch/CUDA version hell:
#   - Unsloth's pip resolver pulls torch 2.12+ from PyPI (needs CUDA 13 driver — nobody has that)
#   - flash-attn install destroys the env even on failure (upgrades torch, removes unsloth)
#   - torchao crashes on torch <2.7
#
# Solution: pin torch via constraints file, never attempt flash-attn.
#
# Usage:
#   cd /trading_management/langgraph
#   bash optimization/qlora/setup_runpod.sh
#
# After setup:
#   source .venv/bin/activate
#   aws configure
#   dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles
#   python optimization/qlora/train_qlora.py --max-steps 3 --max-eval 20 --eval-batch-size 8 --diagnostic-samples 0
#   nohup bash optimization/qlora/run_cloud.sh > optimization/qlora/logs/run_cloud.log 2>&1 &

set -e

echo "============================================================"
echo "  RunPod QLoRA Setup — $(date)"
echo "============================================================"
echo ""

# ─── 0. PATH ───────────────────────────────────────────────────────────────────
for p in /usr/local/cuda/bin /usr/local/bin /opt/pytorch/bin; do
    [ -d "$p" ] && export PATH="$p:$PATH"
done
grep -q '/.local/bin' ~/.bashrc 2>/dev/null || \
    echo 'export PATH="$HOME/.local/bin:/usr/local/cuda/bin:/usr/local/bin:$PATH"' >> ~/.bashrc

# ─── 1. GPU + driver ──────────────────────────────────────────────────────────
echo "[1/6] GPU + NVIDIA driver..."
if ! nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. Is this a GPU pod?"
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
echo "  Driver max CUDA: $DRIVER_CUDA"

CUDA_MAJOR=$(echo "$DRIVER_CUDA" | cut -d. -f1)
CUDA_MINOR=$(echo "$DRIVER_CUDA" | cut -d. -f2)

# Map driver CUDA to the right torch wheel. Only torch versions with matching
# wheels on https://download.pytorch.org/whl/ are listed.
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
    echo "ERROR: CUDA $DRIVER_CUDA too old (need >=12.1). Redeploy on a newer host."
    exit 1
fi
echo "  → torch==${TORCH_VER}+${TORCH_CUDA}"
echo ""

# ─── 2. Python + dev headers ──────────────────────────────────────────────────
echo "[2/6] Python..."
PYTHON=""
for candidate in python3.12 python3.11 python3.10; do
    command -v "$candidate" &>/dev/null && PYTHON="$candidate" && break
done
[ -z "$PYTHON" ] && PYTHON=$(command -v python3)
echo "  $PYTHON ($($PYTHON --version))"

PYVER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if command -v apt-get &>/dev/null; then
    sudo apt-get update -qq 2>/dev/null || true
    sudo apt-get install -y -q "python${PYVER}-dev" 2>/dev/null || \
        sudo apt-get install -y -q python3-dev 2>/dev/null || true
fi
echo ""

# ─── 3. Clean venv + torch (pinned) ──────────────────────────────────────────
echo "[3/6] Venv + torch..."
[ -d .venv ] && rm -rf .venv
$PYTHON -m venv .venv
source .venv/bin/activate
pip install --upgrade pip -q

# Install torch FIRST from the PyTorch wheel index (not PyPI — PyPI's torch 2.12
# bundles CUDA 13 toolkit that no driver supports yet).
pip install "torch==${TORCH_VER}" torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"

# Verify CUDA works before proceeding
python -c "import torch; assert torch.cuda.is_available(), f'torch {torch.__version__} cannot see GPU'" || {
    echo "ERROR: torch ${TORCH_VER}+${TORCH_CUDA} cannot see the GPU. Driver/CUDA mismatch."
    echo "  nvidia-smi says CUDA $DRIVER_CUDA, torch built for ${TORCH_CUDA}."
    exit 1
}
echo "  torch OK: $(python -c "import torch; print(f'{torch.__version__} CUDA={torch.version.cuda} GPU={torch.cuda.get_device_name(0)}')")"
echo ""

# ─── 4. Unsloth (constrained so pip won't upgrade torch) ─────────────────────
echo "[4/6] Unsloth + training stack..."

# Constraints file: prevents pip from upgrading torch to 2.12+ (which needs CUDA 13)
CONSTRAINTS=$(mktemp)
cat > "$CONSTRAINTS" <<CONS
torch==${TORCH_VER}
torchvision
torchaudio
CONS

pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    -c "$CONSTRAINTS"
rm -f "$CONSTRAINTS"

# On torch <2.7, torchao uses APIs that don't exist. Remove it if broken.
if ! python -c "import torchao" 2>/dev/null; then
    echo "  Removing incompatible torchao..."
    pip uninstall torchao -y 2>/dev/null || true
fi

# NOTE: We do NOT attempt flash-attn installation.
# pip install flash-attn --no-build-isolation resolves deps BEFORE building the
# wheel. When the build fails (no nvcc, wrong CUDA), pip has already upgraded
# torch and removed unsloth to "resolve conflicts". This is unfixable with pip.
# Unsloth uses Triton kernels as fallback — ~10-20% slower, still perfectly fine.
echo "  FlashAttention-2: SKIPPED (Triton fallback — reliable, ~10% slower)"
echo ""

# ─── 5. DVC + AWS CLI ────────────────────────────────────────────────────────
echo "[5/6] DVC + AWS CLI + tmux..."
pip install "dvc[s3]" scikit-learn pyyaml tqdm matplotlib httpx -q

export PATH="/usr/local/bin:$HOME/.local/bin:$PATH"
hash -r
if ! command -v aws &>/dev/null; then
    echo "  Installing AWS CLI v2..."
    curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -qo /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install --update 2>/dev/null || \
        /tmp/aws/install --install-dir "$HOME/.local/aws-cli" --bin-dir "$HOME/.local/bin" --update
    rm -rf /tmp/awscliv2.zip /tmp/aws
    hash -r
fi
echo "  aws: $(aws --version 2>/dev/null || echo 'NOT FOUND')"
if aws sts get-caller-identity &>/dev/null; then
    echo "  AWS: $(aws sts get-caller-identity --query 'Arn' --output text)"
else
    echo "  AWS: no creds yet — run 'aws configure' before dvc pull"
fi

command -v tmux &>/dev/null && echo "  tmux: $(tmux -V)" || \
    { sudo apt-get install -y tmux -qq 2>/dev/null && echo "  tmux: installed" || echo "  tmux: N/A"; }
echo ""

# ─── 6. Final verification ───────────────────────────────────────────────────
echo "[6/6] Verifying..."
python -c "
import torch
print(f'  PyTorch:  {torch.__version__}  (CUDA {torch.version.cuda})')
print(f'  GPU:      {torch.cuda.get_device_name(0)}')
print(f'  VRAM:     {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB')
try:
    import torchao; print(f'  torchao:  {torchao.__version__}')
except ImportError:
    print('  torchao:  removed (not needed)')
from unsloth import FastLanguageModel
print('  Unsloth:  OK')
import trl, transformers, peft, bitsandbytes
print(f'  TRL={trl.__version__}  transformers={transformers.__version__}  peft={peft.__version__}  bnb={bitsandbytes.__version__}')
"
echo ""

mkdir -p optimization/qlora/logs optimization/qlora/results

echo "============================================================"
echo "  DONE. Next:"
echo ""
echo "    source .venv/bin/activate"
echo "    aws configure"
echo "    dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles"
echo ""
echo "    # Smoke test:"
echo "    python optimization/qlora/train_qlora.py --max-steps 3 --max-eval 20 --eval-batch-size 8 --diagnostic-samples 0"
echo ""
echo "    # Full run:"
echo "    nohup bash optimization/qlora/run_cloud.sh > optimization/qlora/logs/run_cloud.log 2>&1 &"
echo "    tail -f optimization/qlora/logs/run_cloud.log"
echo "============================================================"
