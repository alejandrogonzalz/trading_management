#!/bin/bash
# QLoRA — single cloud training run (EC2 g6e.xlarge, L40S 48GB)
#
# Trains ONE model on the fixed strict-temporal split and evaluates on a subset
# of the test set. The thesis needs one defensible cloud model, not a config sweep.
#
# Usage (tmux recommended):
#   cd trading_management/langgraph
#   source .venv/bin/activate
#   dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles   # REQUIRED
#   tmux new -s qlora
#   bash optimization/qlora/run_cloud.sh 2>&1 | tee optimization/qlora/logs/run_cloud.log
#   # Ctrl+B, D to detach; tmux attach -t qlora to reattach
#
# Flags explained:
#   --lr 0.00002        Learning rate. 2e-5 is the sweet spot for QLoRA on instruct models.
#   --rank 16           LoRA adapter dimension (~40M trainable params, 0.6% of 7B).
#   --alpha 32          Adapter scaling factor (2×rank = standard default).
#   --epochs 2          Full passes over the 39K training samples. Early stopping may cut short.
#   --batch-size 2      Samples per GPU step. Max that fits L40S without FA2 (batch 4 OOMs).
#   --grad-accum 8      Accumulate 8 mini-batches → effective batch = 16 (stable gradients).
#   --max-eval 1000     Evaluate 1000 test samples (~91/symbol). Full test (8425) = ~40h.
#   --diagnostic-samples 200  Measure train/val accuracy (200 each) for the overfitting gap.
#   --tag qlora_cloud   Names result files and the model output directory.
#   --output-dir ...    Where adapters + GGUF are saved (DVC-tracked for S3 backup).
#   "$@"                Forwards extra flags (e.g. --resume to continue from checkpoint).
#
# Expected timing (g6e.xlarge, WITHOUT FlashAttention-2):
#   Training:  ~12h (4902 steps × 8.7s/step, 2 epochs)
#   Eval:      ~6.5h (1400 generates × 17s: 1000 test + 200 train + 200 val)
#   GGUF:      ~15 min
#   TOTAL:     ~19h worst case, ~16h if early stopping triggers
#   COST:      ~$30-35 on-demand ($1.86/hr)

set -eo pipefail

# Prefix every line with a wall-clock timestamp. tqdm's per-step lines only carry
# RELATIVE time, so without this you can't tell WHEN a step ran (e.g. to pin down
# exactly when a run died). printf '%(...)T' is a bash 4.2+ builtin — no `ts`/
# moreutils dependency. The Python logger's own timestamps still show too.
_ts() { while IFS= read -r line; do printf '[%(%Y-%m-%d %H:%M:%S)T] %s\n' -1 "$line"; done; }

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

# --max-eval 1000: the full test is 8,425 samples × ~17s = ~40h (impractical).
#   1000 samples ≈ ~91 per symbol (11 symbols) — enough for McNemar significance.
#   Total eval: 1000 test + 200 train + 200 val = 1400 generates ≈ ~6.5h.
#   To eval more later: deploy GGUF to Ollama and use `run-backtest` (no GPU needed).
# --diagnostic-samples 200: train/val accuracy probes for the overfitting gap.
python "$SCRIPT_DIR/train_qlora.py" \
    --lr 0.00002 \
    --rank 16 \
    --alpha 32 \
    --epochs 2 \
    --batch-size 2 \
    --grad-accum 8 \
    --max-eval 1000 \
    --diagnostic-samples 200 \
    --tag "$TAG" \
    --output-dir "backtest/data/models/$TAG" \
    "$@" 2>&1 | _ts

echo ""
echo "  DONE — $(date)"
echo "  Result: optimization/qlora/results/qlora_${TAG}.json (+ canonical qlora_optimization.json)"
echo "  Loss curve: optimization/qlora/results/${TAG}_loss_curve.{json,png}"
echo ""
echo "  Backup the model to S3:"
echo "    aws s3 cp --recursive backtest/data/models/$TAG/ s3://trading-management-dvc/models/$TAG/"
