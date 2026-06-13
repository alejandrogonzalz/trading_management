#!/bin/bash
# QLoRA — 3 configs para tesis
# Tiempo: ~9-18h | Costo: ~$20-42
#
# Uso:
#   tmux new -s qlora
#   source .venv/bin/activate
#   bash optimization/qlora/run_3configs.sh
#   # Ctrl+B, D para detach

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"
TRAIN_SCRIPT="$SCRIPT_DIR/train_qlora.py"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# Ensure we run from langgraph/ root (output-dir paths are relative to CWD)
LANGGRAPH_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$LANGGRAPH_ROOT"
echo "  Working dir: $PWD"

# Remove stale result from smoke test
rm -f "$RESULTS_DIR/qlora_optimization.json"

echo "============================================================"
echo "  QLoRA — 3 configs para tesis"
echo "  Started: $(date)"
echo "  GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null)"
echo "============================================================"
echo ""

declare -a NAMES=("config1" "config2" "config3")
declare -a LRS=("0.00002" "0.00005" "0.00001")
declare -a RANKS=("16" "8" "32")
declare -a ALPHAS=("32" "16" "64")
declare -a EPOCHS=("2" "3" "2")

TOTAL=3
PASSED=0
FAILED=0

for i in "${!NAMES[@]}"; do
    NAME="${NAMES[$i]}"
    LR="${LRS[$i]}"
    RANK="${RANKS[$i]}"
    ALPHA="${ALPHAS[$i]}"
    EP="${EPOCHS[$i]}"

    echo "------------------------------------------------------------"
    echo "  [$((i+1))/$TOTAL] $NAME: lr=$LR rank=$RANK alpha=$ALPHA epochs=$EP"
    echo "  Started: $(date)"
    echo "------------------------------------------------------------"

    LOG_FILE="$LOG_DIR/qlora_${NAME}.log"
    START_TIME=$(date +%s)
    OUTPUT_DIR="backtest/data/models/qlora_${NAME}"

    if python "$TRAIN_SCRIPT" \
        --lr "$LR" \
        --rank "$RANK" \
        --alpha "$ALPHA" \
        --epochs "$EP" \
        --batch-size 4 \
        --grad-accum 4 \
        --max-eval 2000 \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_FILE"; then

        RESULT_JSON="$RESULTS_DIR/qlora_optimization.json"
        if [ -f "$RESULT_JSON" ]; then
            RESULT_MTIME=$(stat -c %Y "$RESULT_JSON" 2>/dev/null || stat -f %m "$RESULT_JSON" 2>/dev/null)
            if [ "$RESULT_MTIME" -ge "$START_TIME" ]; then
                cp "$RESULT_JSON" "$RESULTS_DIR/qlora_${NAME}.json"
                PASSED=$((PASSED + 1))
                echo ""
                echo "  $NAME completed. Result saved."
            else
                FAILED=$((FAILED + 1))
                echo "  $NAME: result JSON is stale. Check log."
            fi
        else
            FAILED=$((FAILED + 1))
            echo "  $NAME: no result JSON produced."
        fi
    else
        FAILED=$((FAILED + 1))
        echo "  $NAME FAILED. Check $LOG_FILE"
    fi

    echo "  Finished: $(date)"
    echo ""
done

echo "============================================================"
echo "  DONE — $(date)"
echo "  Passed: $PASSED / $TOTAL    Failed: $FAILED"
echo "============================================================"
echo ""

python3 -c "
import json, glob, os
results = []
for f in sorted(glob.glob('$RESULTS_DIR/qlora_config*.json')):
    try:
        r = json.load(open(f))
        name = os.path.basename(f).replace('.json', '')
        acc = r.get('best_score', 0)
        loss = r.get('val_metrics', {}).get('eval_loss', 999)
        params = r.get('best_params', {})
        results.append((name, acc, loss, params))
    except Exception as e:
        print(f'  ERROR reading {f}: {e}')
if results:
    results.sort(key=lambda x: -x[1])
    print(f'  Config        Accuracy   Eval Loss  LR         Rank')
    print(f'  -----------  ---------  ---------  ---------  ----')
    for name, acc, loss, params in results:
        print(f'  {name:<12} {acc:<10.4f} {loss:<10.4f} {params.get(\"learning_rate\",\"?\"):<10} {params.get(\"lora_rank\",\"?\")}')
    print()
    w = results[0]
    print(f'  Winner: {w[0]} (accuracy={w[1]:.4f})')
"

echo ""
echo "  PARA LA INSTANCIA AHORA — te esta costando ~\$2.35/hr"
echo ""
