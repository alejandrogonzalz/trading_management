#!/bin/bash
# QLoRA — single LOCAL training run (RTX 5070 Ti 16GB), OPTIONAL.
#
# A cheaper second data point on consumer hardware: same fixed strict-temporal
# split, 1 epoch (local is ~19h/epoch at 28s/step, so 1 is the practical max).
# 16GB VRAM forces batch=1 / grad_accum=16.
#
# Linux/WSL/git-bash usage (from langgraph/):
#   source .venv-finetuning/bin/activate
#   dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles   # REQUIRED
#   bash optimization/qlora/run_local.sh 2>&1 | tee optimization/qlora/logs/run_local.log
#
# Native Windows PowerShell equivalent:
#   .\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py `
#     --lr 0.00005 --rank 8 --alpha 16 --epochs 1 --max-steps 1500 `
#     --batch-size 1 --grad-accum 16 --max-eval 200 --diagnostic-samples 0 `
#     --tag qlora_local --output-dir backtest\data\models\qlora_local `
#     2>&1 | Tee-Object -FilePath logs\qlora_local.log
#
# Flags explained:
#   --lr 0.00005        Higher LR (5e-5) — compensates for only 1 epoch (less total updates).
#   --rank 8            Smaller LoRA (~20M params vs 40M in cloud). Fits 16GB comfortably.
#   --alpha 16          2×rank (standard).
#   --epochs 1          One pass, capped at 1500 steps (~60% of epoch).
#   --batch-size 1      Only value that fits 1024-token samples in 16GB.
#   --grad-accum 16     Effective batch = 1×16 = 16 (same as cloud).
#   --max-steps 1500    Cap training at 1500 steps (~60% of 1 epoch). Learns the main
#                       patterns without the full 19h commitment. Resume with --resume
#                       to continue to 2451 if time allows.
#   --max-eval 200      Evaluate 200 test samples (~18/symbol). Light but representative.
#   --diagnostic-samples 0  Skip train/val gap probes (saves ~2h). Run cloud for full diag.
#   --tag qlora_local   Names output files.
#   "$@"                Forwards extra flags (e.g. --resume).
#
# Expected timing (RTX 5070 Ti, no FA2, Triton kernels):
#   Training:  ~11.7h (1500 steps × 28s/step)
#   Eval:      ~2h (200 generates × 35s)
#   GGUF:      ~15 min
#   TOTAL:     ~14h (overnight)
#   COST:      $0 (local hardware)

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
cd "$LANGGRAPH_ROOT"

TAG="qlora_local"
echo "============================================================"
echo "  QLoRA — single local run ($TAG)  [RTX 5070 Ti 16GB]"
echo "  Started: $(date)"
echo "  Working dir: $PWD"
echo "============================================================"

python "$SCRIPT_DIR/train_qlora.py" \
    --lr 0.00005 \
    --rank 8 \
    --alpha 16 \
    --epochs 1 \
    --max-steps 1500 \
    --batch-size 1 \
    --grad-accum 16 \
    --max-eval 200 \
    --diagnostic-samples 0 \
    --tag "$TAG" \
    --output-dir "backtest/data/models/$TAG" \
    "$@" 2>&1 | _ts

echo ""
echo "  DONE — $(date)"
echo "  Result: optimization/qlora/results/$TAG/result.json (canonical: qlora_optimization.json)"
echo "  Loss curve: optimization/qlora/results/$TAG/loss_curve.{json,png}"
