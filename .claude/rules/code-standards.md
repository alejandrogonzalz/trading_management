# Code Standards — Optimization & Backtest Modules

## Architecture Pattern

All ML training/search code follows the class-based pipeline pattern:

```
BaseSearcher (ABC)         → defines .search(cfg, dataset_path) → dict
  ├── SklearnSearcher      → XGBoost, Random Forest (RandomizedSearchCV)
  ├── LSTMSearcher         → Manual loop with early stopping
  ├── QLoRASearcher        → Config generation for Unsloth
  ├── BaggingLSTMSearcher  → N LSTMs with different seeds
  ├── AdaBoostSearcher     → GridSearchCV over stumps
  ├── VotingSearcher       → Weighted probability averaging
  ├── StackingSearcher     → OOF predictions + meta-learner
  └── BlendingSearcher     → Hold-out blend set + meta-learner

OptimizerPipeline          → orchestrates: load config → get searcher → run → save
```

## Adding a New Model

1. Create a searcher class in `optimization/searchers/` extending `BaseSearcher`
2. Implement `.search(cfg, dataset_path) -> dict` returning the standard format:
   ```python
   {
       "model": str,
       "best_score": float,
       "best_params": dict,
       "val_metrics": {"accuracy": ..., "f1_macro": ..., "auc_roc": ...},
       "test_metrics": {"accuracy": ..., "f1_macro": ..., "auc_roc": ...},
       "elapsed_seconds": float,
       "timestamp": str,
   }
   ```
3. Register in `OptimizerPipeline._get_searcher()` or call directly from a run script
4. Create a YAML config in `optimization/configs/` if it has tunable params

## Shared Utilities

- `_DataMixin` in ensemble_searcher.py: `_load_all()`, `_evaluate()`, `_build_lstm_sequences()`
- `backtest/models/features.py`: `_load_dataset()`, `_temporal_split()`, `_samples_to_xy()`
- `optimization/io/results.py`: `save_result()`, `load_all_results()`
- `optimization/io/plots.py`: `save_optimization_plot()`, `save_comparison_plot()`

## Result Format Consistency

Every searcher returns a dict that `save_result()` can serialize to JSON.
The `analyze_results.py` script reads all `*_optimization.json` files and produces a comparison.

## Monitoring

- `optimization/monitor.py` — checks processes, logs, and result files
- Usage: `python optimization/monitor.py --watch 30` (refreshes every 30s)
- Logs go to `langgraph/logs/` (gitignored)

## Running Long Jobs

```bash
# tmux (Cloud Desktop — survives SSH disconnect)
tmux new -s <name>
OMP_NUM_THREADS=4 python optimization/run_ensembles.py 2>&1 | tee logs/ensembles.log
# Ctrl+B, D to detach

# nohup (Mac — add caffeinate to prevent sleep)
caffeinate -dims nohup python optimization/run_ensembles.py > logs/ensembles.log 2>&1 &

# Monitor from another terminal
python optimization/monitor.py --watch 30
```
