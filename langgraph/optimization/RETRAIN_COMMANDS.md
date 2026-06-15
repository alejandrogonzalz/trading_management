# Re-Training ML Models on Fixed Temporal Split

All commands below use the strict temporal holdout (`_temporal_split` with embargo).
Results are valid for paired comparison against the QLoRA model.

## Prerequisites

```powershell
cd C:\Users\alex\projects\trading_management\langgraph
$env:OMP_NUM_THREADS = 1
```

## Individual Models (Hyperparameter Optimization)

### XGBoost (80 random iterations, ~10-15 min)

```powershell
.\.venv\Scripts\python.exe optimization\optimize.py --model xgboost --config optimization\configs\xgboost.yaml
```

### Random Forest (80 random iterations, ~10-15 min)

```powershell
.\.venv\Scripts\python.exe optimization\optimize.py --model random_forest --config optimization\configs\random_forest.yaml
```

### LSTM (30 random iterations, ~30-60 min)

```powershell
.\.venv\Scripts\python.exe optimization\optimize.py --model lstm --config optimization\configs\lstm.yaml
```

## Ensembles (Bagging-LSTM, AdaBoost, Voting, Stacking, Blending)

Runs all 5 ensemble methods sequentially (~2-4 hours total):

```powershell
.\.venv\Scripts\python.exe optimization\run_ensembles.py
```

## Quick Train + Evaluate (CLI — uses best_params from optimization)

```powershell
# XGBoost
.\.venv\Scripts\python.exe -m cli train-ml --dataset backtest\data\labeled\dataset.jsonl --model xgboost --serialize

# LSTM
.\.venv\Scripts\python.exe -m cli train-ml --dataset backtest\data\labeled\dataset.jsonl --model lstm --serialize

# Random Forest
.\.venv\Scripts\python.exe -m cli train-ml --dataset backtest\data\labeled\dataset.jsonl --model random-forest --serialize
```

## Run All (Recommended Order)

```powershell
# 1. Individual models first (they produce best_params for ensembles)
.\.venv\Scripts\python.exe optimization\optimize.py --model xgboost --config optimization\configs\xgboost.yaml
.\.venv\Scripts\python.exe optimization\optimize.py --model random_forest --config optimization\configs\random_forest.yaml
.\.venv\Scripts\python.exe optimization\optimize.py --model lstm --config optimization\configs\lstm.yaml

# 2. Ensembles (uses best params from step 1)
.\.venv\Scripts\python.exe optimization\run_ensembles.py

# 3. Statistical comparison against QLoRA (once both are done)
.\.venv\Scripts\python.exe -m cli compare-stats --a optimization\qlora\results\qlora_optimization.json --b optimization\results\lstm_optimization.json
```

## Output Locations

| What | Path |
|------|------|
| XGBoost results | `optimization/results/xgboost_optimization.json` |
| Random Forest results | `optimization/results/random_forest_optimization.json` |
| LSTM results | `optimization/results/lstm_optimization.json` |
| Ensemble results | `optimization/results/{bagging_lstm,adaboost,...}_optimization.json` |
| Saved models | `backtest/data/models/` |

## Notes

- All models use the same `_temporal_split` (70/15/15, global timestamp sort + embargo)
- Results include `sample_keys` for paired McNemar/t-test comparison
- `--serialize` saves the trained model for later inference
- XGBoost/RF are CPU-only; LSTM uses PyTorch (GPU if available, CPU fine too)
- Ensemble `run_ensembles.py` hardcodes best LSTM/XGB params from Avance 4 — update if optimization finds better ones
