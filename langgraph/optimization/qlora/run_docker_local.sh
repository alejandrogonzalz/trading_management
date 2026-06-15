#!/usr/bin/env bash
# Run INSIDE docker.io/unsloth/unsloth:latest container on RTX 5070 Ti
#
# From langgraph/ on Windows (PowerShell):
#   docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
#     -v "${PWD}:/workspace" -w /workspace \
#     docker.io/unsloth/unsloth:latest \
#     bash optimization/qlora/run_docker_local.sh
#
# From WSL2 (project in WSL2 filesystem for best I/O):
#   docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
#     -v "$(pwd):/workspace" -w /workspace \
#     docker.io/unsloth/unsloth:latest \
#     bash optimization/qlora/run_docker_local.sh
#
# See docs/LOCAL_DOCKER_TRAINING.md for full guide.

set -e

LOG_DIR="optimization/qlora/logs"
mkdir -p "$LOG_DIR"

echo "=== Docker local QLoRA training — RTX 5070 Ti 16GB ==="
echo "=== Check for 'FA2 = True' in the Unsloth startup banner below ==="
echo ""

# ── smoke test (3 steps, 20-sample eval) ─────────────────────────────────────
echo "--- Smoke test (3 steps) ---"
python3 optimization/qlora/train_qlora.py \
  --lr 5e-5 \
  --rank 8 \
  --alpha 16 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 16 \
  --tag qlora_docker_local \
  --max-steps 3 \
  --max-eval 20 \
  2>&1 | tee "$LOG_DIR/smoke_docker_local.log"

echo ""
echo "--- Smoke test passed (no OOM). Starting full 1-epoch run. ---"
echo "--- Monitor: python3 optimization/qlora/monitor_training.py --log $LOG_DIR/run_docker_local.log --epochs 1 --tz -6 --watch ---"
echo ""

# ── full single-epoch run ─────────────────────────────────────────────────────
python3 optimization/qlora/train_qlora.py \
  --lr 5e-5 \
  --rank 8 \
  --alpha 16 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 16 \
  --tag qlora_docker_local \
  2>&1 | tee "$LOG_DIR/run_docker_local.log"

echo ""
echo "=== Completed. Result: optimization/qlora/results/qlora_docker_local.json ==="
