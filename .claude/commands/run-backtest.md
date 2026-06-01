---
description: Run ML backtest with trade simulation on the test set
---

Run `MLBacktestRunner` for a model. Simulates trades with TP/SL on historical candles.

## Usage
`/run-backtest <model_type> [tuned]`

- model_type: `lstm`, `xgboost`, or `random-forest`
- `tuned`: load best_params from optimization/results/ automatically

## Steps

1. Determine model and params:
   - If "tuned": load from `langgraph/optimization/results/{model}_optimization.json`
   - Otherwise: use defaults from `backtest/models/sklearn_models.py` or `lstm.py`

2. Execute:
```bash
cd langgraph && OMP_NUM_THREADS=1 .venv/bin/python -c "
from backtest.evaluation.runner import MLBacktestRunner
runner = MLBacktestRunner(
    dataset_path='backtest/data/labeled/dataset.jsonl',
    model_type='$MODEL',
    tag='$TAG',
    params=$PARAMS,
    serialize=True,
    verbose=False,
)
result = runner.run()
m = result['metrics']
print(f\"Accuracy: {m['direction_accuracy']:.4f}\")
print(f\"Win Rate: {m['win_rate']:.4f}\")
print(f\"Sharpe:   {m['sharpe_ratio']:.2f}\")
print(f\"PF:       {m['profit_factor']:.2f}\")
print(f\"MaxDD:    {m['max_drawdown']:.2f}%\")
"
```

3. Report the key trading metrics to the user.
