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

# Activate venv only when needed (RunPod unsloth image is pre-activated)
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -n "$VIRTUAL_ENV" ] || python3 -c "import torch" 2>/dev/null; then
    : # already in a working environment
else
    echo "ERROR: no .venv found and torch not importable. Activate your venv first." >&2
    exit 1
fi

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
PYTHON=$(command -v python3 || command -v python)

# Ensure PyTorch's bundled cuDNN takes precedence over any system cuDNN in LD_LIBRARY_PATH.
# Without this, a mismatched system cuDNN (e.g. 9.8.0 vs PyTorch's 9.10.2) causes a
# RuntimeError on the first LSTM .to(device) call.
TORCH_LIB=$($PYTHON -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))" 2>/dev/null || true)
if [ -n "$TORCH_LIB" ] && [ -d "$TORCH_LIB" ]; then
    export LD_LIBRARY_PATH="$TORCH_LIB:${LD_LIBRARY_PATH:-}"
fi

echo "[1/3] LSTM grid search..."
$PYTHON optimization/optimize.py \
  --model lstm \
  --config optimization/configs/lstm.yaml \
  --dataset "$DATASET"

echo ""
echo "[2/3] XGBoost grid search..."
$PYTHON optimization/optimize.py \
  --model xgboost \
  --config optimization/configs/xgboost.yaml \
  --dataset "$DATASET"

echo ""
echo "[3/3] Ensembles (using best_params from steps 1-2)..."
if [ -n "$TAG" ]; then
    $PYTHON optimization/run_ensembles.py --force --tag "$TAG"
else
    $PYTHON optimization/run_ensembles.py --force
fi

echo ""
echo "============================================================"
echo "  DONE — $(date)"
echo "============================================================"
