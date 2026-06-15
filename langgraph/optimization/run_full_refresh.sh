#!/bin/bash
# Full refresh: individual models → ensembles (sequential, auto-chains)
#
# Usage:
#   nohup bash optimization/run_full_refresh.sh > logs/full_refresh.log 2>&1 & echo "PID: $!"
#
# With a tag (keeps old results, saves new ones separately):
#   bash optimization/run_full_refresh.sh --tag v2
#
# Without a tag + --force (overwrites old results):
#   bash optimization/run_full_refresh.sh

set -e  # stop on first failure
cd "$(dirname "$0")/.."
source .venv/bin/activate

DATASET="backtest/data/labeled/dataset.jsonl"
export OMP_NUM_THREADS=1

# Parse --tag flag
TAG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag) TAG="$2"; shift 2 ;;
        *) shift ;;
    esac
done

echo "============================================================"
echo "  FULL REFRESH — $(date)"
echo "============================================================"

echo ""
echo "[1/3] LSTM grid search..."
python optimization/optimize.py \
  --model lstm \
  --config optimization/configs/lstm.yaml \
  --dataset "$DATASET"

echo ""
echo "[2/3] XGBoost grid search..."
python optimization/optimize.py \
  --model xgboost \
  --config optimization/configs/xgboost.yaml \
  --dataset "$DATASET"

echo ""
echo "[3/3] Ensembles (using best_params from steps 1-2)..."
if [ -n "$TAG" ]; then
    python optimization/run_ensembles.py --force --tag "$TAG"
else
    python optimization/run_ensembles.py --force
fi

echo ""
echo "============================================================"
echo "  DONE — $(date)"
echo "============================================================"
