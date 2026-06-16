#!/bin/bash
# ==============================================================================
# No-Drawdown-Filter experiment — one-command launcher for a GPU box.
#
# Drives the whole ablation end-to-end with fail-fast preflight checks and a
# smoke test BEFORE the long run, so an agent (or a human) can start it and debug
# failures from structured output instead of chaining four commands by hand.
#
#   Stage 1  Preflight   — python / CUDA GPU / candles present (cheap, seconds)
#   Stage 2  Dataset     — build dataset_no_drawdown_filter.jsonl if missing
#   Stage 3  Smoke test  — 3 training steps on the no_filter path (catches OOM /
#                          import / data errors in ~1 min, before committing hours)
#   Stage 4  Full run    — delegates to run_cloud.sh (GPU batch autodetect, eval,
#                          DVC push, git commit) with the right flags + tag
#
# Usage (from langgraph/, after `dvc pull backtest/data/candles`):
#   bash optimization/qlora/run_experiment_no_drawdown_filter.sh            # full
#   bash optimization/qlora/run_experiment_no_drawdown_filter.sh --smoke-only
#   bash optimization/qlora/run_experiment_no_drawdown_filter.sh --skip-smoke
#   bash optimization/qlora/run_experiment_no_drawdown_filter.sh --skip-checks
#
# Recommended (survive disconnects + keep a log):
#   nohup bash optimization/qlora/run_experiment_no_drawdown_filter.sh \
#     > optimization/qlora/logs/qlora_no_drawdown_filter.log 2>&1 & echo "PID: $!"
#
# See docs: .claude/EXPERIMENT_NO_DRAWDOWN_FILTER.md (plan + Agent Runbook),
#           .claude/RESULT_INTERPRETATION.md (how to read the result).
# ==============================================================================

set -uo pipefail  # NOT -e: every stage handles its own failure with a clear hint

TAG="qlora_no_drawdown_filter"
SYMBOLS="BTCUSDT,ETHUSDT,BNBUSDT,SOLUSDT,XRPUSDT,ADAUSDT,AVAXUSDT,DOTUSDT,DOGEUSDT,LINKUSDT,MATICUSDT,NEARUSDT"
TIMEFRAMES="1h,4h,1d"
LR="0.00002"; RANK="16"; ALPHA="32"; EPOCHS="2"   # match the qlora_cloud baseline

SMOKE_ONLY=0; SKIP_SMOKE=0; SKIP_CHECKS=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --smoke-only)  SMOKE_ONLY=1; shift ;;
        --skip-smoke)  SKIP_SMOKE=1; shift ;;
        --skip-checks) SKIP_CHECKS=1; shift ;;
        *) echo "Unknown flag: $1" >&2; exit 2 ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LANGGRAPH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$LANGGRAPH_ROOT"
DATASET="backtest/data/labeled/dataset_no_drawdown_filter.jsonl"
CANDLES_DIR="backtest/data/candles"
mkdir -p "$SCRIPT_DIR/logs"

step() { echo; echo "============================================================"; echo "  $*"; echo "============================================================"; }
ok()   { echo "  OK   — $*"; }
fail() { echo; echo "  FAIL — $*" >&2; echo "  (see .claude/EXPERIMENT_NO_DRAWDOWN_FILTER.md → Agent Runbook for fixes)" >&2; exit 1; }

# Pick a safe smoke-test batch size from VRAM (the full run autodetects its own).
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
VRAM_MB="${VRAM_MB:-0}"
if   [ "$VRAM_MB" -ge 70000 ]; then SMOKE_BATCH=8
elif [ "$VRAM_MB" -ge 40000 ]; then SMOKE_BATCH=4
else SMOKE_BATCH=2; fi

# ---------------------------------------------------------------- Stage 1: preflight
if [ "$SKIP_CHECKS" -eq 0 ]; then
    step "[1/4] Preflight checks"
    command -v python3 >/dev/null 2>&1 || fail "python3 not found on PATH"
    ok "python3: $(python3 --version 2>&1)"

    python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
        || fail "no CUDA GPU visible to torch (nvidia-smi? launched with --gpus all? right venv?)"
    ok "CUDA GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1) [smoke batch=$SMOKE_BATCH]"

    N_CANDLES=$(ls "$CANDLES_DIR"/*_1h.json 2>/dev/null | wc -l | tr -d ' ')
    [ "$N_CANDLES" -ge 10 ] || fail "only $N_CANDLES *_1h.json candle files in $CANDLES_DIR — run: dvc pull backtest/data/candles"
    ok "candles: $N_CANDLES symbols with 1h data"
else
    echo "  (preflight skipped via --skip-checks)"
fi

# ---------------------------------------------------------------- Stage 2: dataset
step "[2/4] Unfiltered dataset"
if [ -f "$DATASET" ]; then
    ok "exists: $DATASET ($(wc -l < "$DATASET" | tr -d ' ') samples) — reusing"
else
    echo "  building $DATASET (drawdown filter OFF)..."
    python3 -m cli prepare-dataset --symbols "$SYMBOLS" --timeframes "$TIMEFRAMES" --no-drawdown-filter \
        || fail "dataset generation failed (candles present? see log above)"
    [ -f "$DATASET" ] || fail "prepare-dataset finished but $DATASET is missing"
    ok "built: $(wc -l < "$DATASET" | tr -d ' ') samples (filtered baseline is 56,161)"
fi

# ---------------------------------------------------------------- Stage 3: smoke test
if [ "$SKIP_SMOKE" -eq 0 ]; then
    step "[3/4] Smoke test (3 steps, no_filter path)"
    python3 optimization/qlora/train_qlora.py \
        --dataset-type no_filter --max-steps 3 --max-eval 4 --eval-batch-size 4 \
        --diagnostic-samples 0 --batch-size "$SMOKE_BATCH" --no-gguf --tag "${TAG}_smoke" \
        || fail "smoke test failed at batch=$SMOKE_BATCH. If OOM: retry with a smaller --batch-size, or check 'nvidia-smi' for a zombie process holding VRAM."
    ok "smoke test passed — training loop + no_filter data load are healthy"
else
    echo "  (smoke test skipped via --skip-smoke)"
fi

if [ "$SMOKE_ONLY" -eq 1 ]; then
    step "DONE (--smoke-only) — ready for the full run"
    echo "  Launch it with:"
    echo "    nohup bash optimization/qlora/run_experiment_no_drawdown_filter.sh --skip-smoke \\"
    echo "      > optimization/qlora/logs/${TAG}.log 2>&1 & echo \"PID: \$!\""
    exit 0
fi

# ---------------------------------------------------------------- Stage 4: full run
step "[4/4] Full run → run_cloud.sh (tag=$TAG)"
echo "  Config: lr=$LR rank=$RANK alpha=$ALPHA epochs=$EPOCHS dataset-type=no_filter"
echo "  run_cloud.sh handles GPU batch autodetect, eval, GGUF, DVC push + git commit."
bash optimization/qlora/run_cloud.sh \
    --tag "$TAG" --lr "$LR" --rank "$RANK" --alpha "$ALPHA" --epochs "$EPOCHS" \
    --dataset-type no_filter \
    || fail "full run failed — inspect the run_cloud.sh output above"

step "EXPERIMENT COMPLETE"
echo "  Result:  optimization/qlora/results/$TAG/result.json"
echo "  Model:   backtest/data/models/$TAG/"
echo "  Next:    paired compare vs zero-shot on the SAME unfiltered test set —"
echo "           see .claude/EXPERIMENT_NO_DRAWDOWN_FILTER.md §4 step 4, then"
echo "           apply .claude/RESULT_INTERPRETATION.md."
