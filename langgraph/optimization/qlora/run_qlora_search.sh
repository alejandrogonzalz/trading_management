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

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$SCRIPT_DIR/results"
LOG_DIR="$SCRIPT_DIR/logs"
TRAIN_SCRIPT="$SCRIPT_DIR/train_qlora.py"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

echo "============================================================"
echo "  QLoRA Hyperparameter Search — 5 configs × 3 epochs"
echo "  Started: $(date)"
echo "  Instance: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'GPU not detected')"
echo "============================================================"
echo ""

# Define the 5 configs from optimization/configs/qlora.yaml
declare -a NAMES=("config1" "config2" "config3" "config4" "config5")
declare -a LRS=("0.00002" "0.00005" "0.00001" "0.0001" "0.00005")
declare -a RANKS=("16" "8" "32" "16" "16")
declare -a ALPHAS=("32" "16" "64" "32" "32")
declare -a EPOCHS=("3" "3" "3" "1" "3")
declare -a BATCH=("8" "8" "8" "8" "8")

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

    echo "------------------------------------------------------------"
    echo "  [$((i+1))/$TOTAL] $NAME: lr=$LR rank=$RANK alpha=$ALPHA epochs=$EP batch=$BS"
    echo "  Started: $(date)"
    echo "------------------------------------------------------------"

    LOG_FILE="$LOG_DIR/qlora_${NAME}.log"

    if python "$TRAIN_SCRIPT" \
        --lr "$LR" \
        --rank "$RANK" \
        --alpha "$ALPHA" \
        --epochs "$EP" \
        --batch-size "$BS" \
        2>&1 | tee "$LOG_FILE"; then

        # Copy result before next run overwrites it
        cp "$RESULTS_DIR/qlora_optimization.json" "$RESULTS_DIR/qlora_${NAME}.json"
        PASSED=$((PASSED + 1))
        echo ""
        echo "  ✓ $NAME completed. Result saved to qlora_${NAME}.json"
    else
        FAILED=$((FAILED + 1))
        echo ""
        echo "  ✗ $NAME FAILED. Check $LOG_FILE for details."
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
"

echo ""
echo "  Next steps:"
echo "    1. git add optimization/qlora/results/qlora_config*.json"
echo "    2. git commit -m 'feat(qlora): hyperparameter search — 5 configs on SageMaker'"
echo "    3. git push origin dev"
echo "    4. STOP THE INSTANCE (you're paying ~\$2/hr right now)"
echo ""
