#!/usr/bin/env bash
# QLoRA local training — runs INSIDE docker.io/unsloth/unsloth:latest
#
# This script runs INSIDE the container. Launch it via:
#   - docker-compose:  docker compose -f optimization/qlora/docker-compose.local.yml up
#   - PowerShell:      .\optimization\qlora\run_docker_windows.ps1
#   - WSL2 manual:     docker run --rm --gpus all --ipc=host -v "$(pwd):/workspace" -w /workspace \
#                        docker.io/unsloth/unsloth:latest bash optimization/qlora/run_docker_local.sh
#
# Flags (same as run_cloud.sh):
#   --tag NAME     Name for result files and model dir (default: qlora_local)
#   --lr VALUE     Learning rate (default: 5e-5)
#   --rank VALUE   LoRA rank (default: 8)
#   --alpha VALUE  LoRA alpha (default: 16)
#   --epochs N     Number of epochs (default: 1)
#   --resume       Continue from last checkpoint
#   --max-steps N  For smoke tests
#   --max-eval N   Limit eval samples
#   --skip-checks  Skip health checks (use if you already validated)
#
# Examples:
#   bash optimization/qlora/run_docker_local.sh
#   bash optimization/qlora/run_docker_local.sh --tag config2 --lr 0.00005 --rank 8 --epochs 3
#   bash optimization/qlora/run_docker_local.sh --tag config3 --lr 0.00001 --rank 32 --alpha 64 --epochs 2

set -eo pipefail

# ─── Parse flags ─────────────────────────────────────────────────────────────
TAG="qlora_local"
LR="0.00005"
RANK="8"
ALPHA="16"
EPOCHS="1"
SKIP_CHECKS=false
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)          TAG="$2"; shift 2 ;;
        --lr)           LR="$2"; shift 2 ;;
        --rank)         RANK="$2"; shift 2 ;;
        --alpha)        ALPHA="$2"; shift 2 ;;
        --epochs)       EPOCHS="$2"; shift 2 ;;
        --skip-checks)  SKIP_CHECKS=true; shift ;;
        *)              EXTRA_ARGS+=("$1"); shift ;;
    esac
done

LOG_DIR="optimization/qlora/logs"
mkdir -p "$LOG_DIR" optimization/qlora/results

# ─── Health checks ───────────────────────────────────────────────────────────
CHECKS_PASSED=0
CHECKS_FAILED=0
WARNINGS=0

check_pass() { echo "  ✓ $1"; CHECKS_PASSED=$((CHECKS_PASSED+1)); }
check_fail() { echo "  ✗ $1"; CHECKS_FAILED=$((CHECKS_FAILED+1)); }
check_warn() { echo "  ! $1"; WARNINGS=$((WARNINGS+1)); }

if [ "$SKIP_CHECKS" = false ]; then
    echo "============================================================"
    echo "  HEALTH CHECKS"
    echo "============================================================"
    echo ""

    # 1. GPU accessible
    echo "  [GPU]"
    if nvidia-smi &>/dev/null; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')
        VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
        VRAM_FREE=$((VRAM_MB - VRAM_USED))
        check_pass "GPU found: $GPU_NAME (${VRAM_MB} MB total)"

        if [ "$VRAM_USED" -gt 1000 ]; then
            check_warn "GPU has ${VRAM_USED} MB in use — close Ollama/Chrome/games for best results"
        else
            check_pass "GPU memory mostly free (${VRAM_USED} MB used)"
        fi
    else
        check_fail "nvidia-smi not found — Docker not launched with --gpus all?"
    fi
    echo ""

    # 2. Python + core packages
    echo "  [Python Stack]"
    PYTHON=$(command -v python3 || command -v python)
    if [ -z "$PYTHON" ]; then
        check_fail "python3 not found"
    else
        check_pass "Python: $($PYTHON --version)"

        # torch + CUDA
        TORCH_GPU=$($PYTHON -c "import torch; print(torch.cuda.is_available())" 2>/dev/null) || TORCH_GPU="False"
        if [ "$TORCH_GPU" = "True" ]; then
            TORCH_VER=$($PYTHON -c "import torch; print(f'{torch.__version__} CUDA {torch.version.cuda}')")
            check_pass "torch: $TORCH_VER (GPU=True)"
        else
            check_fail "torch cannot see GPU (torch.cuda.is_available()=False)"
        fi

        # unsloth
        if $PYTHON -c "from unsloth import FastLanguageModel" &>/dev/null; then
            check_pass "unsloth: importable"
        else
            check_fail "unsloth import failed"
        fi

        # bitsandbytes
        if $PYTHON -c "import bitsandbytes" &>/dev/null; then
            check_pass "bitsandbytes: OK"
        else
            check_fail "bitsandbytes missing (QLoRA 4-bit requires it)"
        fi

        # flash-attn
        FA2_VER=$($PYTHON -c "import flash_attn; print(flash_attn.__version__)" 2>/dev/null) || FA2_VER=""
        if [ -n "$FA2_VER" ]; then
            check_pass "FlashAttention-2: v$FA2_VER"
        else
            check_warn "FlashAttention-2 not available — will use xformers fallback (~20% slower)"
        fi

        # TRL
        if $PYTHON -c "import trl" &>/dev/null; then
            check_pass "TRL: OK"
        else
            check_fail "TRL missing (SFTTrainer requires it)"
        fi
    fi
    echo ""

    # 3. Project dependencies
    echo "  [Project Deps]"
    MISSING_DEPS=""
    $PYTHON -c "import sklearn" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS scikit-learn"
    $PYTHON -c "import xgboost" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS xgboost"
    $PYTHON -c "import yaml" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS pyyaml"
    $PYTHON -c "import tqdm" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS tqdm"
    $PYTHON -c "import matplotlib" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS matplotlib"
    $PYTHON -c "import httpx" 2>/dev/null || MISSING_DEPS="$MISSING_DEPS httpx"

    if [ -z "$MISSING_DEPS" ]; then
        check_pass "All project deps installed"
    else
        check_warn "Missing (will install now):$MISSING_DEPS"
        pip install $MISSING_DEPS -q 2>&1 | tail -2
        check_pass "Installed:$MISSING_DEPS"
    fi
    echo ""

    # 4. Data files
    echo "  [Data]"
    DATASET="backtest/data/labeled/dataset.jsonl"
    if [ -f "$DATASET" ]; then
        LINES=$(wc -l < "$DATASET")
        if [ "$LINES" -ge 50000 ]; then
            check_pass "Dataset: $LINES samples"
        else
            check_warn "Dataset has only $LINES samples (expected ~56161)"
        fi
    else
        check_fail "Dataset not found at $DATASET — run 'dvc pull' on the HOST first"
    fi

    CANDLE_COUNT=$(ls backtest/data/candles/*.json 2>/dev/null | wc -l)
    if [ "$CANDLE_COUNT" -ge 30 ]; then
        check_pass "Candles: $CANDLE_COUNT files"
    elif [ "$CANDLE_COUNT" -gt 0 ]; then
        check_warn "Only $CANDLE_COUNT candle files (expected 36) — some symbols may lack trade simulation"
    else
        check_warn "No candle files — financial metrics (win_rate, profit_factor) will be 0"
    fi
    echo ""

    # 5. Disk space
    echo "  [Disk]"
    AVAIL_MB=$(df -m /workspace 2>/dev/null | tail -1 | awk '{print $4}')
    if [ -n "$AVAIL_MB" ] && [ "$AVAIL_MB" -ge 20000 ]; then
        check_pass "Disk: ${AVAIL_MB} MB free (need ~15 GB for checkpoints + GGUF)"
    elif [ -n "$AVAIL_MB" ]; then
        check_warn "Disk: only ${AVAIL_MB} MB free — may run out during GGUF export (needs ~15 GB)"
    fi
    echo ""

    # 6. Summary
    echo "────────────────────────────────────────────────────────────"
    echo "  Results: $CHECKS_PASSED passed, $CHECKS_FAILED failed, $WARNINGS warnings"
    echo "────────────────────────────────────────────────────────────"

    if [ "$CHECKS_FAILED" -gt 0 ]; then
        echo ""
        echo "  ABORTING — $CHECKS_FAILED critical check(s) failed."
        echo "  Fix the issues above and re-run."
        echo ""
        echo "  Common fixes:"
        echo "    GPU not found     → docker run --gpus all (check NVIDIA Container Toolkit)"
        echo "    torch no GPU      → image is stale: docker pull docker.io/unsloth/unsloth:latest"
        echo "    Dataset missing   → run 'dvc pull' on the HOST (not inside the container)"
        echo "    unsloth fail      → wrong image or corrupted pull: docker rmi + re-pull"
        exit 1
    fi
    echo ""
fi

# ─── Auto-detect batch size ──────────────────────────────────────────────────
VRAM_MB=${VRAM_MB:-$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')}
VRAM_MB="${VRAM_MB:-0}"

if [ "$VRAM_MB" -ge 70000 ]; then
    BATCH=8; GRAD_ACCUM=2; EVAL_BATCH=32
elif [ "$VRAM_MB" -ge 40000 ]; then
    BATCH=2; GRAD_ACCUM=8; EVAL_BATCH=16
elif [ "$VRAM_MB" -ge 20000 ]; then
    BATCH=2; GRAD_ACCUM=8; EVAL_BATCH=8
else
    BATCH=1; GRAD_ACCUM=16; EVAL_BATCH=4
fi

# ─── Training ────────────────────────────────────────────────────────────────
echo "============================================================"
echo "  QLoRA Docker Local — $TAG"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "  VRAM: ${VRAM_MB} MB → BATCH=$BATCH GRAD_ACCUM=$GRAD_ACCUM (effective $((BATCH * GRAD_ACCUM)))"
echo "  Config: lr=$LR rank=$RANK alpha=$ALPHA epochs=$EPOCHS"
echo "  Tag: $TAG"
echo "  Started: $(date)"
echo "============================================================"

# Smoke test — catch OOM before committing to a long run
echo ""
echo "--- Smoke test (3 steps, 20-sample eval) ---"
if ! python3 optimization/qlora/train_qlora.py \
  --lr "$LR" --rank "$RANK" --alpha "$ALPHA" \
  --epochs 1 --batch-size "$BATCH" --grad-accum "$GRAD_ACCUM" \
  --tag "${TAG}_smoke" --max-steps 3 --max-eval 20 --no-gguf \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/smoke_${TAG}.log"; then
    echo ""
    echo "  SMOKE TEST FAILED. Check $LOG_DIR/smoke_${TAG}.log"
    echo "  Common causes:"
    echo "    - OOM: close other GPU processes (Ollama, Chrome), or reduce batch_size"
    echo "    - Dataset not found: dvc pull on the host"
    echo "    - Import error: image may need pulling"
    exit 1
fi

# Clean smoke test artifacts
rm -rf "backtest/data/models/${TAG}_smoke" 2>/dev/null || true
rm -f "optimization/qlora/results/qlora_${TAG}_smoke.json" 2>/dev/null || true

echo ""
echo "--- Smoke passed. Starting full run ($EPOCHS epoch(s)). ---"
echo "--- Estimated time: $([ "$VRAM_MB" -ge 70000 ] && echo '~2-3h' || echo '~14-19h') ---"
echo ""

python3 optimization/qlora/train_qlora.py \
  --lr "$LR" --rank "$RANK" --alpha "$ALPHA" \
  --epochs "$EPOCHS" --batch-size "$BATCH" --grad-accum "$GRAD_ACCUM" \
  --eval-batch-size "$EVAL_BATCH" \
  --diagnostic-samples 200 \
  --tag "$TAG" --output-dir "backtest/data/models/$TAG" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "$LOG_DIR/run_${TAG}.log"

echo ""
echo "============================================================"
echo "  COMPLETE — $(date)"
echo ""
echo "  Results:"
echo "    Metrics:  optimization/qlora/results/qlora_${TAG}.json"
echo "    Model:    backtest/data/models/$TAG/"
echo "    Log:      $LOG_DIR/run_${TAG}.log"
echo ""
echo "  Next steps:"
echo "    1. Compare: python3 -m cli compare-stats --a optimization/qlora/results/qlora_${TAG}.json --b optimization/qlora/results/qlora_qlora_cloud.json"
echo "    2. Deploy:  ollama create trading-qwen-ft -f backtest/data/models/$TAG/Modelfile"
echo "============================================================"
