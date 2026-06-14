#!/bin/bash
# QLoRA — single cloud training run (SageMaker ml.g6e.xlarge, L40S 48GB)
#
# Trains ONE model on the fixed strict-temporal split and evaluates on the FULL
# test set (no --max-eval cap → the test spans multiple symbols, not just LINK).
# The thesis needs one defensible cloud model, not a config sweep.
#
# Usage:
#   cd trading_management/langgraph
#   source .venv/bin/activate
#   dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles   # REQUIRED
#   nohup bash optimization/qlora/run_cloud.sh > optimization/qlora/logs/run_cloud.log 2>&1 &
#   echo "PID: $!"
#
# Best config from the audit: lr=2e-5, rank=16, alpha=32, epochs=3, batch=2, grad_accum=8.
# Expected: ~1-3h/epoch on L40S with FA2, ~$6-12 total.

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LANGGRAPH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"

# output-dir paths are relative to CWD → run from langgraph/ root.
cd "$LANGGRAPH_ROOT"

TAG="qlora_cloud"
echo "============================================================"
echo "  QLoRA — single cloud run ($TAG)"
echo "  Started: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'GPU not detected')"
echo "  Working dir: $PWD"
echo "============================================================"

# No --max-eval: evaluate the COMPLETE test split (representative, multi-symbol).
# --diagnostic-samples 500: train/val accuracy probes for the overfitting gap.
python "$SCRIPT_DIR/train_qlora.py" \
    --lr 0.00002 \
    --rank 16 \
    --alpha 32 \
    --epochs 3 \
    --batch-size 2 \
    --grad-accum 8 \
    --diagnostic-samples 500 \
    --tag "$TAG" \
    --output-dir "backtest/data/models/$TAG" \
    "$@"

echo ""
echo "  DONE — $(date)"
echo "  Result: optimization/qlora/results/qlora_${TAG}.json (+ canonical qlora_optimization.json)"
echo "  Loss curve: optimization/qlora/results/${TAG}_loss_curve.{json,png}"
echo ""
echo "  Backup the model to S3:"
echo "    aws s3 cp --recursive backtest/data/models/$TAG/ s3://trading-management-dvc/models/$TAG/"
