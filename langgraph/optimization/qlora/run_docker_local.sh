#!/usr/bin/env bash
# QLoRA local training — runs INSIDE docker.io/unsloth/unsloth:latest
#
# Launch from langgraph/ (WSL2 recommended, PowerShell also works):
#
#   WSL2:
#     docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
#       -v "$(pwd):/workspace" -w /workspace \
#       docker.io/unsloth/unsloth:latest \
#       bash optimization/qlora/run_docker_local.sh
#
#   PowerShell:
#     docker run --rm --gpus all --ipc=host --ulimit memlock=-1 `
#       -v "${PWD}:/workspace" -w /workspace `
#       docker.io/unsloth/unsloth:latest `
#       bash optimization/qlora/run_docker_local.sh
#
# Pass extra flags after the image name to override defaults, e.g.:
#   ... docker.io/unsloth/unsloth:latest bash optimization/qlora/run_docker_local.sh --max-steps 3 --max-eval 20

set -eo pipefail

TAG="qlora_local"
LOG_DIR="optimization/qlora/logs"
mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  QLoRA local — docker.io/unsloth/unsloth:latest"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'not detected')"
echo "  Check startup banner for: FA [FA2 = True]"
echo "============================================================"

# Smoke test first — catch OOM before committing to an overnight run
echo ""
echo "--- Smoke test (3 steps, 20-sample eval) ---"
python3 optimization/qlora/train_qlora.py \
  --lr 5e-5 --rank 8 --alpha 16 \
  --epochs 1 --batch-size 1 --grad-accum 16 \
  --tag "$TAG" --max-steps 3 --max-eval 20 \
  "$@" \
  2>&1 | tee "$LOG_DIR/smoke_local.log"

echo ""
echo "--- Smoke passed. Starting full 1-epoch run (~14h). ---"
echo "--- Monitor (new terminal): python3 optimization/qlora/monitor_training.py --log $LOG_DIR/run_local.log --epochs 1 --watch ---"
echo ""

python3 optimization/qlora/train_qlora.py \
  --lr 5e-5 --rank 8 --alpha 16 \
  --epochs 1 --batch-size 1 --grad-accum 16 \
  --max-eval 200 --diagnostic-samples 0 \
  --tag "$TAG" --output-dir "backtest/data/models/$TAG" \
  "$@" \
  2>&1 | tee "$LOG_DIR/run_local.log"

echo ""
echo "============================================================"
echo "  Done. Result: optimization/qlora/results/$TAG/result.json"
echo "============================================================"
