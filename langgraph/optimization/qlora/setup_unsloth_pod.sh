#!/bin/bash
# RunPod setup for the official unsloth/unsloth:latest image.
#
# This image ships torch + unsloth + flash-attn + bitsandbytes pre-compiled as
# a consistent unit. Our script ONLY validates the pre-installed stack and adds
# the few extra packages we need (DVC, sklearn, xgboost, AWS CLI).
#
# Unlike setup_runpod.sh (which builds everything from scratch on a bare torch
# image), this script assumes the heavy lifting is already done and focuses on
# project-specific setup.
#
# Usage:
#   # After pod starts, SSH in or open a terminal:
#   cd /workspace/work
#   git clone <repo-url> trading_management && cd trading_management/langgraph
#   bash optimization/qlora/setup_unsloth_pod.sh
#
# RunPod template settings:
#   Image:          docker.io/unsloth/unsloth:latest
#   GPU:            A100 80GB (or H100, L40S, RTX 4090 — any Ampere+)
#   Container Disk: 100 GB (image is ~25-30 GB uncompressed + model weights + checkpoints)
#   Volume Disk:    100-256 GB (mounted at /workspace, persists across restarts)
#   HTTP Port:      8888 (JupyterLab)
#   Start Command:  (leave blank — image auto-starts Jupyter)
#
# Image ships: torch 2.10.0+cu128, unsloth, bitsandbytes, TRL>=0.18, peft,
#              xformers (attention fallback), triton, vLLM, nvcc+gcc+ninja.
# Does NOT ship: flash-attn (we build it from source, ~10-30 min with cached wheel).
# Container user: `unsloth` — set --user root in RunPod if permission issues arise.

set -e

echo "============================================================"
echo "  Unsloth Pod Setup — $(date)"
echo "============================================================"
echo ""

# ─── 0. Detect working directory ─────────────────────────────────────────────
# Script should be run from langgraph/ root. Auto-detect if run from elsewhere.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LANGGRAPH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$LANGGRAPH_ROOT"
echo "  Working dir: $PWD"
echo ""

# ─── 1. Validate GPU + driver ────────────────────────────────────────────────
echo "[1/7] GPU validation..."
if ! nvidia-smi &>/dev/null; then
    echo "  ERROR: nvidia-smi not found. Is this a GPU pod?"
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
DRIVER_CUDA=$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1)
echo "  GPU:          $GPU_NAME"
echo "  VRAM:         $((VRAM_MB / 1024)) GB"
echo "  Driver CUDA:  $DRIVER_CUDA"

# Detect actual CUDA toolkit (nvcc) — RunPod uses forward compat so toolkit
# version can be HIGHER than what nvidia-smi reports.
if command -v nvcc &>/dev/null; then
    TOOLKIT_CUDA=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+')
    echo "  Toolkit CUDA: $TOOLKIT_CUDA (from nvcc)"
else
    TOOLKIT_CUDA="$DRIVER_CUDA"
    echo "  Toolkit CUDA: $TOOLKIT_CUDA (nvcc not found, using driver value)"
fi
echo ""

# ─── 2. Validate pre-installed stack ─────────────────────────────────────────
echo "[2/7] Validating pre-installed stack..."

# Find the right python — prefer the image's base python (not a stale venv)
PYTHON=$(command -v python3 || command -v python)
echo "  Python: $($PYTHON --version) at $(which $PYTHON)"

ERRORS=0

# torch
TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__)" 2>/dev/null) || {
    echo "  ERROR: torch not found"; ERRORS=$((ERRORS+1)); TORCH_VER="MISSING"
}
TORCH_CUDA=$($PYTHON -c "import torch; print(torch.version.cuda or 'cpu')" 2>/dev/null) || TORCH_CUDA="?"
TORCH_GPU=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null) || TORCH_GPU="False"
echo "  torch:        $TORCH_VER (CUDA $TORCH_CUDA, GPU=$TORCH_GPU)"

if [ "$TORCH_GPU" != "True" ]; then
    echo "  ERROR: torch cannot see the GPU!"
    ERRORS=$((ERRORS+1))
fi

# unsloth
UNSLOTH_OK=$($PYTHON -c "from unsloth import FastLanguageModel; print('OK')" 2>/dev/null) || UNSLOTH_OK="FAIL"
echo "  unsloth:      $UNSLOTH_OK"
[ "$UNSLOTH_OK" != "OK" ] && ERRORS=$((ERRORS+1))

# flash-attn
FA2_VER=$($PYTHON -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null) || FA2_VER="NOT INSTALLED"
echo "  flash-attn:   $FA2_VER"

# bitsandbytes
BNB_VER=$($PYTHON -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null) || BNB_VER="NOT INSTALLED"
echo "  bitsandbytes: $BNB_VER"
[ "$BNB_VER" = "NOT INSTALLED" ] && ERRORS=$((ERRORS+1))

# TRL + transformers + peft
TRL_VER=$($PYTHON -c "import trl; print(trl.__version__)" 2>/dev/null) || TRL_VER="NOT INSTALLED"
TF_VER=$($PYTHON -c "import transformers; print(transformers.__version__)" 2>/dev/null) || TF_VER="NOT INSTALLED"
PEFT_VER=$($PYTHON -c "import peft; print(peft.__version__)" 2>/dev/null) || PEFT_VER="NOT INSTALLED"
echo "  trl:          $TRL_VER"
echo "  transformers: $TF_VER"
echo "  peft:         $PEFT_VER"
[ "$TRL_VER" = "NOT INSTALLED" ] && ERRORS=$((ERRORS+1))

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "  WARNING: $ERRORS critical packages missing or broken."
    echo "  This doesn't look like the official unsloth/unsloth image."
    echo "  Consider using setup_runpod.sh instead (full from-scratch setup)."
    echo ""
    read -p "  Continue anyway? [y/N] " -n 1 -r
    echo ""
    [[ ! $REPLY =~ ^[Yy]$ ]] && exit 1
fi
echo ""

# ─── 3. Install missing packages (constraints protect torch) ─────────────────
echo "[3/7] Installing project dependencies..."

# Constraints file: even on a pre-built image, a careless `pip install` can
# upgrade torch if a transitive dep requests a newer version. Pin it.
CONSTRAINTS=$(mktemp)
cat > "$CONSTRAINTS" <<CONS
torch==$(echo "$TORCH_VER" | cut -d'+' -f1)
CONS
export PIP_CONSTRAINT="$CONSTRAINTS"

# Only install what's not already there
INSTALL_LIST=""
$PYTHON -c "import dvc" 2>/dev/null || INSTALL_LIST="$INSTALL_LIST dvc[s3]"
$PYTHON -c "import sklearn" 2>/dev/null || INSTALL_LIST="$INSTALL_LIST scikit-learn"
$PYTHON -c "import xgboost" 2>/dev/null || INSTALL_LIST="$INSTALL_LIST xgboost"
$PYTHON -c "import yaml" 2>/dev/null || INSTALL_LIST="$INSTALL_LIST pyyaml"
$PYTHON -c "import tqdm" 2>/dev/null || INSTALL_LIST="$INSTALL_LIST tqdm"
$PYTHON -c "import matplotlib" 2>/dev/null || INSTALL_LIST="$INSTALL_LIST matplotlib"
$PYTHON -c "import httpx" 2>/dev/null || INSTALL_LIST="$INSTALL_LIST httpx"
$PYTHON -c "import einops" 2>/dev/null || INSTALL_LIST="$INSTALL_LIST einops"

if [ -n "$INSTALL_LIST" ]; then
    echo "  Installing:$INSTALL_LIST"
    pip install $INSTALL_LIST -q
else
    echo "  All project dependencies already present."
fi

unset PIP_CONSTRAINT
rm -f "$CONSTRAINTS"

# Verify torch didn't get swapped
FINAL_TORCH=$($PYTHON -c "import torch; print(torch.__version__)")
if [ "$FINAL_TORCH" != "$TORCH_VER" ]; then
    echo "  ERROR: torch changed from $TORCH_VER to $FINAL_TORCH after install!"
    echo "  This is the exact problem we're trying to avoid. Aborting."
    exit 1
fi
echo ""

# ─── 4. Flash-Attn (install if missing, build from source) ───────────────────
echo "[4/7] FlashAttention-2..."

if [ "$FA2_VER" != "NOT INSTALLED" ]; then
    echo "  Already installed: v$FA2_VER"
else
    echo "  Not pre-installed (image uses xformers as fallback)."
    echo "  Building from source for ~10-20% attention speedup..."

    # Check for nvcc (required for source build — should be present in unsloth image)
    if ! command -v nvcc &>/dev/null; then
        echo "  WARNING: nvcc not found. Cannot compile flash-attn."
        echo "  Training will use xformers fallback (still works fine)."
    else
        # Determine GPU arch for targeted compilation
        SM_TAG=$($PYTHON -c "import torch; cc=torch.cuda.get_device_capability(0); print(f'{cc[0]}.{cc[1]}')")
        CPU_CORES=$(nproc 2>/dev/null || echo 4)
        MAX_JOBS=$(( CPU_CORES > 16 ? 16 : CPU_CORES ))

        echo "  Building for sm_$SM_TAG with MAX_JOBS=$MAX_JOBS (~10-15 min)..."

        # Check S3 cache first
        FA_LATEST=$(pip index versions flash-attn 2>/dev/null | grep -oP '(?<=flash-attn \()[^)]+' | head -1 || echo "2.8.3")
        PY_TAG=$($PYTHON -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
        TORCH_CUDA_TAG=$(echo "$TORCH_CUDA" | tr -d '.')
        FA_WHEEL_NAME="flash_attn-${FA_LATEST}-cu${TORCH_CUDA_TAG}-${PY_TAG}-sm${SM_TAG/./}-linux_x86_64.whl"
        FA_S3="s3://trading-management-dvc/wheels/${FA_WHEEL_NAME}"
        FA_LOCAL="/tmp/fa_wheel/${FA_WHEEL_NAME}"
        mkdir -p /tmp/fa_wheel

        FA_OK=false

        # Try S3 cache
        if command -v aws &>/dev/null && aws s3 cp "$FA_S3" "$FA_LOCAL" 2>/dev/null; then
            echo "  Found cached wheel in S3 — installing..."
            pip install "$FA_LOCAL" --no-deps -q
            if $PYTHON -c "import flash_attn" 2>/dev/null; then
                FA_OK=true
                echo "  FA2: OK (from S3 cache)"
            else
                pip uninstall flash-attn -y 2>/dev/null || true
            fi
        fi

        # Source build
        if ! $FA_OK; then
            WHEEL_DIR="/tmp/fa_wheel"
            if FLASH_ATTENTION_FORCE_BUILD=TRUE \
               TORCH_CUDA_ARCH_LIST="$SM_TAG" \
               CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" \
               MAX_JOBS=$MAX_JOBS \
               pip wheel flash-attn --no-build-isolation --no-deps -w "$WHEEL_DIR" 2>&1 | tail -5; then

                BUILT_WHEEL=$(ls "$WHEEL_DIR"/flash_attn-*.whl 2>/dev/null | head -1)
                if [ -n "$BUILT_WHEEL" ] && pip install "$BUILT_WHEEL" --no-deps -q; then
                    if $PYTHON -c "import flash_attn" 2>/dev/null; then
                        FA_OK=true
                        echo "  FA2: OK (built from source)"
                        # Cache to S3 for next pod
                        if command -v aws &>/dev/null; then
                            aws s3 cp "$BUILT_WHEEL" "$FA_S3" 2>/dev/null && \
                                echo "  Cached to S3: $FA_S3" || true
                        fi
                    fi
                fi
            fi
        fi

        if ! $FA_OK; then
            echo "  FA2: FAILED — training will use xformers fallback (still works, ~10-20% slower)"
        fi
    fi
fi
echo ""

# ─── 5. AWS CLI ──────────────────────────────────────────────────────────────
echo "[5/7] AWS CLI..."
export PATH="$PATH:$HOME/.local/bin:/usr/local/bin"

if ! command -v aws &>/dev/null; then
    echo "  Installing AWS CLI v2..."
    curl -sL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
    unzip -qo /tmp/awscliv2.zip -d /tmp
    sudo /tmp/aws/install --update 2>/dev/null || \
        /tmp/aws/install --install-dir "$HOME/.local/aws-cli" --bin-dir "$HOME/.local/bin" --update
    rm -rf /tmp/awscliv2.zip /tmp/aws
    hash -r
fi

if aws sts get-caller-identity &>/dev/null; then
    echo "  AWS: $(aws sts get-caller-identity --query 'Arn' --output text)"
else
    echo "  AWS CLI installed but no credentials configured."
    echo "  Run 'aws configure' before dvc pull."
fi
echo ""

# ─── 6. DVC pull (if AWS configured) ─────────────────────────────────────────
echo "[6/7] DVC data..."
if aws sts get-caller-identity &>/dev/null; then
    if [ ! -f backtest/data/labeled/dataset.jsonl ]; then
        echo "  Pulling dataset + candles from S3..."
        dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles 2>&1 | tail -3
    else
        echo "  Dataset already present ($(wc -l < backtest/data/labeled/dataset.jsonl) lines)"
    fi

    # Verify candles exist (needed for financial metrics)
    CANDLE_COUNT=$(ls backtest/data/candles/*.json 2>/dev/null | wc -l)
    if [ "$CANDLE_COUNT" -eq 0 ]; then
        echo "  WARNING: No candle files found. Financial metrics will be 0."
        echo "  Run: dvc pull backtest/data/candles"
    else
        echo "  Candles: $CANDLE_COUNT files"
    fi
else
    echo "  Skipped (no AWS credentials). Run after 'aws configure':"
    echo "    dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles"
fi
echo ""

# ─── 7. Smoke test ───────────────────────────────────────────────────────────
echo "[7/7] Smoke test..."
mkdir -p optimization/qlora/logs optimization/qlora/results

# Quick 3-step training test to verify the full pipeline works
echo "  Running 3-step smoke test (should take <30s)..."
if $PYTHON optimization/qlora/train_qlora.py \
    --max-steps 3 --max-eval 5 --eval-batch-size 1 --diagnostic-samples 0 \
    --batch-size 2 --tag smoke_test 2>&1 | grep -E "(FA|step|loss|Error|OOM)" | head -20; then
    echo "  Smoke test PASSED"
    # Clean up smoke test artifacts
    rm -rf backtest/data/models/smoke_test 2>/dev/null || true
    rm -f optimization/qlora/results/qlora_smoke_test.json 2>/dev/null || true
else
    echo "  Smoke test FAILED — check output above"
    echo "  Common fixes:"
    echo "    - OOM: reduce --batch-size to 1"
    echo "    - Import error: check torch/unsloth versions above"
    echo "    - Dataset not found: run dvc pull first"
fi
echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────
FA2_STATUS=$($PYTHON -c "import flash_attn; print('YES')" 2>/dev/null || echo "NO")
echo "============================================================"
echo "  SETUP COMPLETE — $(date)"
echo ""
echo "  Stack:"
echo "    torch:      $TORCH_VER (CUDA $TORCH_CUDA)"
echo "    GPU:        $GPU_NAME ($((VRAM_MB / 1024)) GB)"
echo "    FA2:        $FA2_STATUS"
echo "    unsloth:    OK"
echo ""

# Recommend batch size based on GPU + FA2
if [ "$VRAM_MB" -ge 70000 ] && [ "$FA2_STATUS" = "YES" ]; then
    REC_BATCH=8; REC_GA=2; REC_SPEED="~1.5-2s/step (~2-3h total)"
elif [ "$VRAM_MB" -ge 70000 ]; then
    REC_BATCH=2; REC_GA=8; REC_SPEED="~4-5s/step (~6h total)"
elif [ "$VRAM_MB" -ge 40000 ] && [ "$FA2_STATUS" = "YES" ]; then
    REC_BATCH=4; REC_GA=4; REC_SPEED="~3-4s/step (~4-5h total)"
elif [ "$VRAM_MB" -ge 40000 ]; then
    REC_BATCH=2; REC_GA=8; REC_SPEED="~5-7s/step (~8h total)"
else
    REC_BATCH=1; REC_GA=16; REC_SPEED="~8-12s/step (~12h+ total)"
fi

echo "  Recommended: BATCH=$REC_BATCH GRAD_ACCUM=$REC_GA → $REC_SPEED"
echo ""
echo "  To start training:"
echo "    bash optimization/qlora/run_cloud.sh 2>&1 | tee optimization/qlora/logs/run_cloud.log"
echo ""
echo "  To run ML grid search in parallel (CPU only):"
echo "    OMP_NUM_THREADS=4 python optimization/optimize.py --model xgboost --config optimization/configs/xgboost.yaml > optimization/logs/xgboost.log 2>&1 &"
echo "    OMP_NUM_THREADS=4 python optimization/optimize.py --model random_forest --config optimization/configs/random_forest.yaml > optimization/logs/rf.log 2>&1 &"
echo "    CUDA_VISIBLE_DEVICES=\"\" python optimization/optimize.py --model lstm --config optimization/configs/lstm.yaml > optimization/logs/lstm.log 2>&1 &"
echo "============================================================"
