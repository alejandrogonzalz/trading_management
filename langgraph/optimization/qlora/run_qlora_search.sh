#!/bin/bash
# QLoRA hyperparameter search — runs all 5 recommended configs from qlora.yaml
# sequentially on SageMaker (or any GPU machine).
#
# Usage:
#   cd trading_management/langgraph
#   source .venv/bin/activate
#   bash optimization/qlora/run_qlora_search.sh
#
# Expected time: ~15-30h total on ml.g6e.xlarge (L40S 48GB)
# Expected cost: ~$30-60

set -eo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"
TRAIN_SCRIPT="$SCRIPT_DIR/train_qlora.py"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "============================================================"
echo "  QLoRA Hyperparameter Search — 5 configs"
echo "  Started: $(date)"
echo "  Instance: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'GPU not detected')"
echo "============================================================"
echo ""

# Define the 5 configs from optimization/configs/qlora.yaml
# Each config gets its own output_dir to avoid overwriting adapters (R5 fix).
# grad_accum=2 with batch=8 → effective batch=16 (matches local 1×16, R6 fix).
# --max-eval 2000 for the search (±2% CI, sufficient to rank); run full eval on winner.
declare -a NAMES=("config1" "config2" "config3" "config4" "config5")
declare -a LRS=("0.00002" "0.00005" "0.00001" "0.0001" "0.00005")
declare -a RANKS=("16" "8" "32" "16" "16")
declare -a ALPHAS=("32" "16" "64" "32" "32")
declare -a EPOCHS=("2" "3" "2" "1" "2")
declare -a BATCH=("8" "8" "8" "8" "8")
declare -a GRAD_ACCUM=("2" "2" "2" "2" "2")

TOTAL=${#NAMES[@]}
PASSED=0
FAILED=0

for i in "${!NAMES[@]}"; do
    NAME="${NAMES[$i]}"
    LR="${LRS[$i]}"
    RANK="${RANKS[$i]}"
    ALPHA="${ALPHAS[$i]}"
    EP="${EPOCHS[$i]}"
    BS="${BATCH[$i]}"
    GA="${GRAD_ACCUM[$i]}"

    echo "------------------------------------------------------------"
    echo "  [$((i+1))/$TOTAL] $NAME: lr=$LR rank=$RANK alpha=$ALPHA epochs=$EP batch=$BS grad_accum=$GA"
    echo "  Started: $(date)"
    echo "------------------------------------------------------------"

    LOG_FILE="$LOG_DIR/qlora_${NAME}.log"
    START_TIME=$(date +%s)

    # Each config writes to its own output_dir (avoids overwriting adapters)
    OUTPUT_DIR="backtest/data/models/qlora_${NAME}"

    # Run training. Capture exit code properly despite tee (pipefail is set).
    if python "$TRAIN_SCRIPT" \
        --lr "$LR" \
        --rank "$RANK" \
        --alpha "$ALPHA" \
        --epochs "$EP" \
        --batch-size "$BS" \
        --grad-accum "$GA" \
        --max-eval 2000 \
        --output-dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_FILE"; then

        # Verify the result JSON was actually written (not stale from a previous run)
        RESULT_JSON="$RESULTS_DIR/qlora_optimization.json"
        if [ -f "$RESULT_JSON" ]; then
            RESULT_MTIME=$(stat -c %Y "$RESULT_JSON" 2>/dev/null || stat -f %m "$RESULT_JSON" 2>/dev/null)
            if [ "$RESULT_MTIME" -ge "$START_TIME" ]; then
                cp "$RESULT_JSON" "$RESULTS_DIR/qlora_${NAME}.json"
                PASSED=$((PASSED + 1))
                echo ""
                echo "  ✓ $NAME completed. Result saved to qlora_${NAME}.json"
            else
                FAILED=$((FAILED + 1))
                echo ""
                echo "  ✗ $NAME: result JSON is stale (from a previous run). Check $LOG_FILE"
            fi
        else
            FAILED=$((FAILED + 1))
            echo ""
            echo "  ✗ $NAME: no result JSON produced. Check $LOG_FILE"
        fi
    else
        FAILED=$((FAILED + 1))
        echo ""
        echo "  ✗ $NAME FAILED (exit code $?). Check $LOG_FILE for details."
    fi

    echo "  Finished: $(date)"
    echo ""
done

# Print summary
echo "============================================================"
echo "  SEARCH COMPLETE — $(date)"
echo "  Passed: $PASSED / $TOTAL    Failed: $FAILED"
echo "============================================================"
echo ""
echo "  Results:"
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

if not results:
    print('  No results found.')
else:
    results.sort(key=lambda x: -x[1])  # sort by accuracy descending
    print(f'  {\"Config\":<12} {\"Accuracy\":<10} {\"Eval Loss\":<10} {\"LR\":<10} {\"Rank\":<6} {\"Alpha\":<6}')
    print(f'  {\"-\"*12} {\"-\"*10} {\"-\"*10} {\"-\"*10} {\"-\"*6} {\"-\"*6}')
    for name, acc, loss, params in results:
        lr = params.get('learning_rate', '?')
        rank = params.get('lora_rank', '?')
        alpha = params.get('lora_alpha', '?')
        print(f'  {name:<12} {acc:<10.4f} {loss:<10.4f} {lr:<10} {rank:<6} {alpha:<6}')
    print()
    winner = results[0]
    print(f'  ★ Winner: {winner[0]} (accuracy={winner[1]:.4f}, eval_loss={winner[2]:.4f})')
    print(f'    Params: lr={winner[3].get(\"learning_rate\")}, rank={winner[3].get(\"lora_rank\")}, alpha={winner[3].get(\"lora_alpha\")}')
    print()
    print(f'  To run full eval (8,425 samples) on the winner:')
    print(f'    python optimization/qlora/train_qlora.py --lr {winner[3].get(\"learning_rate\")} --rank {winner[3].get(\"lora_rank\")} --alpha {winner[3].get(\"lora_alpha\")} --epochs {winner[3].get(\"epochs\", 3)} --batch-size 8 --grad-accum 2')
"

echo ""
echo "  Next steps:"
echo "    1. Run full eval on the winner (without --max-eval) if needed"
echo "    2. git add optimization/qlora/results/qlora_config*.json"
echo "    3. git commit -m 'feat(qlora): hyperparameter search — 5 configs on SageMaker'"
echo "    4. git push origin dev"
echo "    5. STOP THE INSTANCE (you're paying ~\$2/hr right now)"
echo ""
