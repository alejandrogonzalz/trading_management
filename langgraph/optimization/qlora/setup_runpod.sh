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

# ─── 4. Unsloth + all deps (single constrained install) ──────────────────────
echo "[4/6] Unsloth + training stack + utilities..."

# Constraints file: prevents ANY pip install from upgrading torch off the pinned
# version. Without this, transitive deps (dvc → s3fs → aiobotocore → ...) can
# trigger pip to pull torch from PyPI (2.12+cu13, incompatible with any driver).
CONSTRAINTS=$(mktemp)
cat > "$CONSTRAINTS" <<CONS
torch==${TORCH_VER}
torchvision
torchaudio
CONS
export PIP_CONSTRAINT="$CONSTRAINTS"

# Install everything in one pass so pip resolves all deps together
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git" \
    "dvc[s3]" scikit-learn pyyaml tqdm matplotlib httpx

unset PIP_CONSTRAINT
rm -f "$CONSTRAINTS"

# On torch <2.7, torchao uses APIs that don't exist. Remove it if broken.
if ! python -c "import torchao" 2>/dev/null; then
    echo "  Removing incompatible torchao..."
    pip uninstall torchao -y 2>/dev/null || true
fi

# Verify torch didn't get swapped
FINAL_TORCH=$(python -c "import torch; print(torch.__version__)")
if [[ "$FINAL_TORCH" != *"${TORCH_CUDA}"* ]] && [[ "$FINAL_TORCH" != "${TORCH_VER}"* ]]; then
    echo "  ERROR: torch changed to $FINAL_TORCH after install! Re-pinning..."
    pip install --force-reinstall "torch==${TORCH_VER}" torchvision torchaudio \
        --index-url "https://download.pytorch.org/whl/${TORCH_CUDA}"
    pip install --force-reinstall "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
    python -c "import torchao" 2>/dev/null || pip uninstall torchao -y 2>/dev/null || true
fi

# FlashAttention-2: source-build with S3 wheel cache.
#
# WHY NOT the prebuilt wheel from GitHub?
#   flash-attn 2.8+ setup.py auto-downloads a prebuilt wheel tagged "cxx11abiFALSE"
#   but those wheels were built against conda PyTorch (CXX11 ABI = True). Our pip-
#   installed PyTorch uses the old ABI (std::string → "Ss" mangling). Result:
#   "undefined symbol: c10::Error(SourceLocation, std::__cxx11::string)" at import.
#   FLASH_ATTENTION_FORCE_BUILD=TRUE bypasses the download and actually compiles.
#
# WHY S3 cache?
#   Source compilation takes ~15 min. We cache the compiled wheel keyed on
#   (flash_attn version, torch version, CUDA tag, Python ABI, GPU SM arch) so
#   every pod after the first gets a 30-second install instead.
#
# Cache key format: flash_attn-{FA_VER}-{TORCH_CUDA}-cp{PY}-sm{CC}-linux_x86_64.whl
# Cache location:   s3://trading-management-dvc/wheels/
FA_VER=$(pip index versions flash-attn 2>/dev/null | grep -oP '(?<=flash-attn \()[^)]+' | head -1 || echo "2.8.3")
PY_TAG=$(python -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
SM_TAG=$(python -c "import torch; cc=torch.cuda.get_device_capability(0); print(f'sm{cc[0]}{cc[1]}')")
FA_WHEEL_NAME="flash_attn-${FA_VER}-${TORCH_CUDA}-${PY_TAG}-${SM_TAG}-linux_x86_64.whl"
FA_S3="s3://trading-management-dvc/wheels/${FA_WHEEL_NAME}"
FA_LOCAL="/tmp/fa_wheel/${FA_WHEEL_NAME}"
mkdir -p /tmp/fa_wheel

echo "  FlashAttention-2 (FA_VER=${FA_VER}, cache key=${FA_WHEEL_NAME})..."
FA_OK=false

# 1. Remove any broken prior install first (import-test; uninstall if broken)
if python -c "import flash_attn" 2>/dev/null; then
    FA_OK=true
    echo "  FlashAttention-2: already installed and working"
else
    pip uninstall flash-attn -y 2>/dev/null || true
fi

# 2. Try S3 cache (instant install — skip ~15 min compile)
if ! $FA_OK && aws s3 cp "$FA_S3" "$FA_LOCAL" 2>/dev/null; then
    echo "  Found cached wheel in S3 — installing..."
    pip install "$FA_LOCAL" --no-deps -q
    if python -c "import flash_attn" 2>/dev/null; then
        FA_OK=true
        echo "  FlashAttention-2: OK (from S3 cache — $(python -c 'import flash_attn; print(flash_attn.__version__)'))"
    else
        echo "  Cached wheel failed to import — rebuilding from source..."
        pip uninstall flash-attn -y 2>/dev/null || true
    fi
fi

# 3. Source build (cache miss or cached wheel was stale)
if ! $FA_OK; then
    echo "  Building from source (~15-20 min). FLASH_ATTENTION_FORCE_BUILD=TRUE skips"
    echo "  the broken prebuilt-wheel download and actually compiles against our torch."
    WHEEL_DIR="/tmp/fa_wheel"
    if FLASH_ATTENTION_FORCE_BUILD=TRUE CUDA_HOME=/usr/local/cuda MAX_JOBS=4 \
            pip wheel flash-attn --no-build-isolation --no-deps -w "$WHEEL_DIR" 2>&1; then
        BUILT_WHEEL=$(ls "$WHEEL_DIR"/flash_attn-*.whl 2>/dev/null | head -1)
        if [ -n "$BUILT_WHEEL" ] && pip install "$BUILT_WHEEL" --no-deps -q; then
            if python -c "import flash_attn" 2>/dev/null; then
                FA_OK=true
                echo "  FlashAttention-2: OK (built from source — $(python -c 'import flash_attn; print(flash_attn.__version__)'))"
                # Upload to S3 for future pods — non-fatal if creds not configured yet
                aws s3 cp "$BUILT_WHEEL" "$FA_S3" 2>/dev/null && \
                    echo "  Cached wheel to S3: $FA_S3" || \
                    echo "  NOTE: wheel not cached (no AWS creds yet). After 'aws configure', run:"
                    echo "        aws s3 cp $BUILT_WHEEL $FA_S3"
            fi
        fi
    fi
fi

if ! $FA_OK; then
    echo "  FlashAttention-2: SKIPPED — Triton fallback active (~20% slower, still fine)"
fi

pip install einops -q 2>/dev/null || true
echo ""

# ─── 5. AWS CLI ──────────────────────────────────────────────────────────────
echo "[5/6] AWS CLI + tmux..."

# APPEND (not prepend) so /usr/local/bin doesn't shadow the venv's python
export PATH="$PATH:$HOME/.local/bin"
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
