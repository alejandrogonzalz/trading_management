#!/bin/bash
# Docker local setup — runs INSIDE docker.io/unsloth/unsloth:latest
#
# Validates the GPU + pre-installed stack, installs the few project deps
# not included in the unsloth image, and runs a smoke test.
#
# This is the Docker equivalent of setup_unsloth_pod.sh but MUCH lighter:
# - No AWS CLI (data is bind-mounted from the host)
# - No DVC (pull happens on the host before launching the container)
# - No llama.cpp pre-build (train_qlora.py handles GGUF export)
# - No persistent volume management
#
# Usage (from langgraph/ as the bind-mount root):
#   docker run --rm --gpus all --ipc=host -v "$(pwd):/workspace" -w /workspace \
#     docker.io/unsloth/unsloth:latest bash optimization/qlora/setup_docker_local.sh
#
# Or use it as a validation step before training:
#   docker run --rm --gpus all --ipc=host -v "$(pwd):/workspace" -w /workspace \
#     docker.io/unsloth/unsloth:latest bash -c \
#     "bash optimization/qlora/setup_docker_local.sh && bash optimization/qlora/run_docker_local.sh"

set -e

echo "============================================================"
echo "  Docker Local Setup — $(date)"
echo "============================================================"
echo ""

# ─── 1. GPU validation ──────────────────────────────────────────────────────
echo "[1/4] GPU validation..."
if ! nvidia-smi &>/dev/null; then
    echo "  ERROR: nvidia-smi not found."
    echo "  Make sure you ran docker with --gpus all"
    echo "  And that NVIDIA Container Toolkit is installed on the host."
    exit 1
fi
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "  GPU:  $GPU_NAME"
echo "  VRAM: $((VRAM_MB / 1024)) GB"
echo ""

# ─── 2. Validate pre-installed stack ────────────────────────────────────────
echo "[2/4] Validating pre-installed stack..."
PYTHON=$(command -v python3 || command -v python)

TORCH_VER=$($PYTHON -c "import torch; print(torch.__version__)" 2>/dev/null) || TORCH_VER="MISSING"
TORCH_GPU=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null) || TORCH_GPU="False"
FA2_VER=$($PYTHON -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null) || FA2_VER="NOT INSTALLED"
BNB_VER=$($PYTHON -c "import bitsandbytes; print(bitsandbytes.__version__)" 2>/dev/null) || BNB_VER="MISSING"
TRL_VER=$($PYTHON -c "import trl; print(trl.__version__)" 2>/dev/null) || TRL_VER="MISSING"
UNSLOTH_OK=$($PYTHON -c "from unsloth import FastLanguageModel; print('OK')" 2>/dev/null) || UNSLOTH_OK="FAIL"

echo "  torch:        $TORCH_VER (GPU=$TORCH_GPU)"
echo "  flash-attn:   $FA2_VER"
echo "  bitsandbytes: $BNB_VER"
echo "  trl:          $TRL_VER"
echo "  unsloth:      $UNSLOTH_OK"

ERRORS=0
[ "$TORCH_GPU" != "True" ] && echo "  ERROR: torch cannot see GPU!" && ERRORS=$((ERRORS+1))
[ "$UNSLOTH_OK" != "OK" ] && echo "  ERROR: unsloth import failed!" && ERRORS=$((ERRORS+1))
[ "$BNB_VER" = "MISSING" ] && echo "  ERROR: bitsandbytes missing!" && ERRORS=$((ERRORS+1))

if [ "$ERRORS" -gt 0 ]; then
    echo ""
    echo "  $ERRORS critical errors. Image may be corrupt or outdated."
    echo "  Try: docker pull docker.io/unsloth/unsloth:latest"
    exit 1
fi

if [ "$FA2_VER" = "NOT INSTALLED" ]; then
    echo "  WARNING: FlashAttention-2 not available. Training will use xformers fallback."
    echo "  This is ~10-20% slower but produces identical results."
fi
echo ""

# ─── 3. Install project dependencies ────────────────────────────────────────
echo "[3/4] Installing project dependencies..."

INSTALL_LIST=""
$PYTHON -c "import sklearn" 2>/dev/null || INSTALL_LIST="$INSTALL_LIST scikit-learn"
$PYTHON -c "import xgboost" 2>/dev/null || INSTALL_LIST="$INSTALL_LIST xgboost"
$PYTHON -c "import yaml" 2>/dev/null || INSTALL_LIST="$INSTALL_LIST pyyaml"
$PYTHON -c "import tqdm" 2>/dev/null || INSTALL_LIST="$INSTALL_LIST tqdm"
$PYTHON -c "import matplotlib" 2>/dev/null || INSTALL_LIST="$INSTALL_LIST matplotlib"
$PYTHON -c "import httpx" 2>/dev/null || INSTALL_LIST="$INSTALL_LIST httpx"

if [ -n "$INSTALL_LIST" ]; then
    echo "  Installing:$INSTALL_LIST"
    pip install $INSTALL_LIST -q 2>&1 | tail -3
else
    echo "  All dependencies already present."
fi
echo ""

# ─── 4. Verify data is available ────────────────────────────────────────────
echo "[4/4] Checking data..."
DATASET="backtest/data/labeled/dataset.jsonl"
if [ -f "$DATASET" ]; then
    LINES=$(wc -l < "$DATASET")
    echo "  Dataset: $LINES samples ✓"
else
    echo "  WARNING: $DATASET not found!"
    echo "  Run 'dvc pull' on the HOST before launching the container."
    echo "  Without it, training will fail."
fi

CANDLE_COUNT=$(ls backtest/data/candles/*.json 2>/dev/null | wc -l)
if [ "$CANDLE_COUNT" -gt 0 ]; then
    echo "  Candles: $CANDLE_COUNT files ✓"
else
    echo "  WARNING: No candle files. Financial metrics (win_rate, profit_factor) will be 0."
    echo "  Run 'dvc pull backtest/data/candles' on the HOST."
fi
echo ""

# ─── Summary ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  SETUP OK — ready to train"
echo ""
echo "  GPU:  $GPU_NAME ($((VRAM_MB / 1024)) GB)"
echo "  FA2:  $([ "$FA2_VER" != "NOT INSTALLED" ] && echo "YES ($FA2_VER)" || echo "NO (xformers fallback)")"
echo "  Data: $([ -f "$DATASET" ] && echo "OK ($LINES samples)" || echo "MISSING")"
echo ""
if [ "$VRAM_MB" -ge 70000 ]; then
    echo "  Recommended: BATCH=8 GRAD_ACCUM=2 (~1.5-2s/step)"
elif [ "$VRAM_MB" -ge 40000 ]; then
    echo "  Recommended: BATCH=2 GRAD_ACCUM=8 (~5-7s/step)"
elif [ "$VRAM_MB" -ge 20000 ]; then
    echo "  Recommended: BATCH=2 GRAD_ACCUM=8 (~15-25s/step)"
else
    echo "  Recommended: BATCH=1 GRAD_ACCUM=16 (~25-35s/step)"
fi
echo ""
echo "  To train: bash optimization/qlora/run_docker_local.sh"
echo "============================================================"
