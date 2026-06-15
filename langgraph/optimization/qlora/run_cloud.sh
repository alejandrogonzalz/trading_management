#!/bin/bash
# QLoRA — single cloud training run (RunPod / EC2 / SageMaker GPU)
#
# Trains ONE model on the fixed strict-temporal split and evaluates a strided,
# representative slice of the test set.
#
# Usage (tmux recommended; nohup on SageMaker Studio):
#   cd trading_management/langgraph
#   source .venv/bin/activate
#   dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles   # REQUIRED
#   tmux new -s qlora
#   bash optimization/qlora/run_cloud.sh 2>&1 | tee optimization/qlora/logs/run_cloud.log
#
# Run with different configs (pass flags BEFORE the -- separator):
#   bash optimization/qlora/run_cloud.sh --tag config2 --lr 0.00005 --rank 8 --epochs 3
#   bash optimization/qlora/run_cloud.sh --tag config3 --lr 0.00001 --rank 32 --epochs 2
#   bash optimization/qlora/run_cloud.sh --tag config1_resume --resume
#
# Tune for the instance via env vars (defaults are SAFE for a 48GB L40S w/o FA2):
#   BATCH=8 GRAD_ACCUM=2 EVAL_BATCH=32 MAX_EVAL=3000 bash optimization/qlora/run_cloud.sh
#   - On a RunPod A100/H100 80GB with FA2: BATCH=8 GRAD_ACCUM=2 (keeps effective
#     batch 16) trains ~4-6x faster; EVAL_BATCH=32 decodes eval in minutes.
#   - Smoke-test a new instance FIRST: append `--max-steps 3 --max-eval 20` to be
#     sure the chosen BATCH/EVAL_BATCH don't OOM before committing to the long run.
#
# Flags explained:
#   --tag NAME          Names result files and model output dir. Default: qlora_cloud.
#   --lr 0.00002        Learning rate. 2e-5 is the sweet spot for QLoRA on instruct models.
#   --rank 16           LoRA adapter dimension (~40M trainable params, 0.6% of 7B).
#   --alpha 32          Adapter scaling factor (2×rank = standard default).
#   --epochs 2          Full passes over the 39K training samples. Early stopping may cut short.
#   --batch-size $BATCH GPU samples per step. 2 fits L40S w/o FA2; 8 on 80GB + FA2.
#   --grad-accum $GA    Mini-batches accumulated → effective batch = BATCH×GA (keep =16).
#   --max-eval $MAX_EVAL  Test samples to eval, STRIDED across the full holdout
#                       (representative, not a prefix). Set to 0/empty to eval all 8425.
#   --eval-batch-size $EVAL_BATCH  Prompts per generate() call. 16 safe on 48GB, 32 on 80GB.
#   --diagnostic-samples 200  Measure train/val accuracy (200 each) for the overfitting gap.
#   --output-dir ...    Where adapters + GGUF are saved (DVC-tracked for S3 backup).
#   --resume            Continue from the latest checkpoint in output-dir.
#   Any extra flags are forwarded to train_qlora.py.
#
# Expected timing (defaults vary by GPU):
#   Training:  ~2-3h on A100/H100 (FA2, BATCH=8); ~12h on L40S (no FA2, BATCH=2)
#   Eval:      ~10-30 min (3000 strided test + 400 diag, batched)
#   GGUF:      ~15 min
#   TOTAL:     ~3-4h on A100/H100; COST ~$5-13 ($1.39 A100 → $3.29 H100 per hr)

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

# Parse script-level flags (consumed here, not forwarded to train_qlora.py)
TAG="qlora_cloud"
LR="0.00002"
RANK="16"
ALPHA="32"
EPOCHS="2"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --tag)      TAG="$2"; shift 2 ;;
        --lr)       LR="$2"; shift 2 ;;
        --rank)     RANK="$2"; shift 2 ;;
        --alpha)    ALPHA="$2"; shift 2 ;;
        --epochs)   EPOCHS="$2"; shift 2 ;;
        *)          EXTRA_ARGS+=("$1"); shift ;;
    esac
done
# Auto-detect GPU VRAM and set optimal batch sizes.
# Override with env vars if needed: BATCH=4 GRAD_ACCUM=4 bash run_cloud.sh
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
VRAM_MB="${VRAM_MB:-0}"

if [ -z "${BATCH+x}" ]; then
    # Check if FlashAttention-2 is available (changes optimal batch size)
    FA2=$(python -c "import flash_attn; print('yes')" 2>/dev/null || echo "no")
    if [ "$VRAM_MB" -ge 70000 ] && [ "$FA2" = "yes" ]; then
        # A100/H100 80GB WITH FA2 — linear attention, batch=8 is fast
        BATCH=8; GRAD_ACCUM=2; EVAL_BATCH=32
    elif [ "$VRAM_MB" -ge 70000 ]; then
        # A100/H100 80GB WITHOUT FA2 — quadratic attention in padding-free mode
        # batch=2 empirically fastest (~4.4s/step vs 6.4s at batch=4, 11s at batch=8)
        BATCH=2; GRAD_ACCUM=8; EVAL_BATCH=32
    elif [ "$VRAM_MB" -ge 40000 ]; then
        # L40S 48GB / A6000 48GB
        BATCH=2; GRAD_ACCUM=8; EVAL_BATCH=16
    else
        # 24GB cards (A10G, RTX 3090/4090)
        BATCH=1; GRAD_ACCUM=16; EVAL_BATCH=8
    fi
else
    GRAD_ACCUM="${GRAD_ACCUM:-$((16 / BATCH))}"
    EVAL_BATCH="${EVAL_BATCH:-16}"
fi
MAX_EVAL="${MAX_EVAL:-3000}"
echo "============================================================"
echo "  QLoRA — cloud run: $TAG"
echo "  Started: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'GPU not detected')"
echo "  Working dir: $PWD"
echo "  Config: lr=$LR rank=$RANK alpha=$ALPHA epochs=$EPOCHS"
echo "  Batch:  BATCH=$BATCH GRAD_ACCUM=$GRAD_ACCUM (effective $((BATCH * GRAD_ACCUM))) | MAX_EVAL=$MAX_EVAL EVAL_BATCH=$EVAL_BATCH"
echo "============================================================"

# --max-eval is now STRIDED across the full 8,425-sample test holdout, so the
#   subset spans the whole period + all symbols (representative, not a prefix).
#   With --eval-batch-size the eval is minutes, not hours, so a larger, more
#   representative slice is cheap. To eval the FULL test, set MAX_EVAL= (empty)
#   or drop the flag. (You can also eval more later via GGUF → Ollama + run-backtest.)
# --diagnostic-samples 200: train/val accuracy probes for the overfitting gap.
EVAL_FLAG=(--eval-batch-size "$EVAL_BATCH")
[ -n "$MAX_EVAL" ] && EVAL_FLAG+=(--max-eval "$MAX_EVAL")
python "$SCRIPT_DIR/train_qlora.py" \
    --lr "$LR" \
    --rank "$RANK" \
    --alpha "$ALPHA" \
    --epochs "$EPOCHS" \
    --batch-size "$BATCH" \
    --grad-accum "$GRAD_ACCUM" \
    "${EVAL_FLAG[@]}" \
    --diagnostic-samples 200 \
    --tag "$TAG" \
    --output-dir "backtest/data/models/$TAG" \
    "${EXTRA_ARGS[@]}" 2>&1 | _ts

echo ""
echo "  DONE — $(date)"
echo "  Result: optimization/qlora/results/$TAG/result.json (canonical: qlora_optimization.json)"
echo "  Loss curve: optimization/qlora/results/$TAG/loss_curve.{json,png}"
echo ""
echo "============================================================"
echo "  Saving results to git + DVC..."
echo "============================================================"

# Go to repo root (DVC root is one level above langgraph/)
REPO_ROOT="$(cd "$LANGGRAPH_ROOT/.." && pwd)"
cd "$REPO_ROOT"

# 1. DVC-track the model weights and push to S3
dvc add "langgraph/backtest/data/models/$TAG/"
dvc push
echo "  [dvc] model weights pushed to s3://trading-management-dvc/"

# 2. Stage and commit everything: DVC pointer + result JSONs + loss curves
git add \
  "langgraph/backtest/data/models/$TAG.dvc" \
  "langgraph/backtest/data/models/.gitignore" \
  "langgraph/optimization/qlora/results/$TAG/result.json" \
  "langgraph/optimization/qlora/results/$TAG/loss_curve.json" \
  "langgraph/optimization/qlora/results/$TAG/loss_curve.png" \
  "langgraph/optimization/qlora/results/qlora_optimization.json" \
  2>/dev/null || true

git commit -m "feat(qlora): add $TAG results and DVC model pointer

$(python3 -c "
import json, sys
try:
    r = json.load(open('langgraph/optimization/qlora/results/$TAG/result.json'))
    m = r.get('metrics', r)
    print(f'  Test acc: {m.get(\"direction_accuracy\", m.get(\"test_accuracy\", \"?\")):.4f}')
    print(f'  Win rate: {m.get(\"win_rate\", \"?\"):.4f}  Profit factor: {m.get(\"profit_factor\", \"?\"):.3f}')
except Exception as e:
    print(f'  (metrics unavailable: {e})')
" 2>/dev/null || echo "  (metrics unavailable)")

Model: backtest/data/models/$TAG/ (DVC-tracked, weights in S3)

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" 2>/dev/null || echo "  [git] nothing new to commit"

# 3. Push commits to remote
git push
echo "  [git] commits pushed to origin"
echo ""
echo "  All done. Model in S3, results in git."
