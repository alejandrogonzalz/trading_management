#!/bin/bash
# QLoRA — single LOCAL training run (RTX 5070 Ti 16GB), OPTIONAL.
#
# A cheaper second data point on consumer hardware: same fixed strict-temporal
# split, full test evaluation, 1 epoch (local is ~15-19h/epoch, so 1 is the
# practical pass). 16GB forces batch=1 / grad_accum=16 and max_seq_length=1024.
#
# Linux/WSL/git-bash usage (from langgraph/):
#   source .venv-finetuning/bin/activate
#   dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles   # REQUIRED
#   bash optimization/qlora/run_local.sh
#
# Native Windows PowerShell equivalent (one line):
#   .\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py `
#     --lr 0.00005 --rank 8 --alpha 16 --epochs 1 --batch-size 1 --grad-accum 16 `
#     --diagnostic-samples 500 --tag qlora_local --output-dir backtest\data\models\qlora_local `
#     2>&1 | Tee-Object -FilePath logs\qlora_local.log

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LANGGRAPH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
LOG_DIR="$SCRIPT_DIR/logs"
mkdir -p "$LOG_DIR"
cd "$LANGGRAPH_ROOT"

TAG="qlora_local"
echo "============================================================"
echo "  QLoRA — single local run ($TAG)  [RTX 5070 Ti 16GB]"
echo "  Started: $(date)"
echo "  Working dir: $PWD"
echo "============================================================"

# max_seq_length stays at the default 1024 (NEVER lower — samples are 877-933 tokens).
python "$SCRIPT_DIR/train_qlora.py" \
    --lr 0.00005 \
    --rank 8 \
    --alpha 16 \
    --epochs 1 \
    --batch-size 1 \
    --grad-accum 16 \
    --diagnostic-samples 500 \
    --tag "$TAG" \
    --output-dir "backtest/data/models/$TAG" \
    "$@"

echo ""
echo "  DONE — $(date)"
echo "  Result: optimization/qlora/results/qlora_${TAG}.json (+ canonical qlora_optimization.json)"
