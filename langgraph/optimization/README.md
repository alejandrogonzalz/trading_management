# Hyperparameter Optimization

Optimization infrastructure for the four trading signal prediction models.

## Structure

```
optimization/
├── pipeline.py            ← OptimizerPipeline orchestrator
├── searchers/
│   ├── base.py            ← BaseSearcher ABC
│   ├── sklearn_searcher.py ← GridSearchCV/RandomizedSearchCV (XGBoost, RF)
│   ├── lstm_searcher.py   ← Manual loop with early stopping
│   └── qlora_searcher.py  ← Config printer for Unsloth manual training
├── io/
│   ├── results.py         ← save_result / load_all_results
│   └── plots.py           ← save_optimization_plot / save_comparison_plot
├── configs/
│   ├── xgboost.yaml       ← Round 2: reg_alpha, reg_lambda, max_depth ≤ 6
│   ├── random_forest.yaml ← Round 2: max_depth ≤ 10, min_samples_leaf ≥ 10
│   ├── lstm.yaml
│   └── qlora.yaml         ← QLoRA hyperparameter search space
├── qlora/                 ← QLoRA fine-tuning (self-contained)
│   ├── train_qlora.py     ← Full pipeline: data → train → eval → save
│   ├── sagemaker_setup.sh ← One-command SageMaker env setup
│   ├── run_qlora_search.sh ← Runs all 5 configs from qlora.yaml
│   ├── SAGEMAKER_GUIDE.md ← Step-by-step SageMaker walkthrough
│   ├── README.md          ← QLoRA-specific docs
│   ├── results/           ← Per-config result JSONs
│   └── logs/              ← Training logs
├── optimize.py            ← CLI entry point
└── analyze_results.py     ← Cross-model comparison
```

## Models

| Model | Searcher | Time estimate | Infrastructure |
|-------|----------|---------------|----------------|
| XGBoost | RandomizedSearchCV | 15-30 min | CPU |
| Random Forest | RandomizedSearchCV | 10-20 min | CPU |
| LSTM | Manual loop + early stopping | 2-6 h | GPU recommended |
| QLoRA (Qwen 2.5 7B) | Manual (Unsloth) | 4-12 h | GPU 16GB+ VRAM |

**Dataset**: ~56K samples, 70/15/15 temporal split.

## Usage

```bash
cd langgraph/optimization

# XGBoost — round 2 random search
python optimize.py --model xgboost --config configs/xgboost.yaml

# Random Forest — round 2 random search
python optimize.py --model random_forest --config configs/random_forest.yaml

# LSTM — manual loop, 30 configs
python optimize.py --model lstm --config configs/lstm.yaml

# QLoRA — prints configs for manual Unsloth training
python optimize.py --model qlora --config configs/qlora.yaml

# Background (long runs)
nohup python optimize.py --model lstm --config configs/lstm.yaml > logs/lstm.log 2>&1 &
tail -f logs/lstm.log

# Compare all results
python analyze_results.py
```

## Output format

Each run writes `results/{model}_optimization.json`:

```json
{
  "model": "xgboost",
  "best_params": {"n_estimators": 300, "max_depth": 4, "...": "..."},
  "best_score": 0.768,
  "all_results": [{"params": {}, "mean_score": 0.75, "rank": 2}],
  "elapsed_seconds": 900,
  "dataset_size": 47737
}
```

`MLBacktestRunner` auto-loads `best_params` from this file when instantiated without explicit `params=`.

## Programmatic use

```python
from optimization.pipeline import OptimizerPipeline

pipeline = OptimizerPipeline(
    model="xgboost",
    config_path="configs/xgboost.yaml",
    dataset_path="../backtest/data/labeled/dataset.jsonl",
)
result = pipeline.run()
```

---

## QLoRA Fine-Tuning

All QLoRA-specific code, scripts, and docs live in [`qlora/`](qlora/README.md).

```bash
# SageMaker (recommended):
bash optimization/qlora/sagemaker_setup.sh
bash optimization/qlora/run_qlora_search.sh

# Local (Windows):
.venv-finetuning\Scripts\python optimization\qlora\train_qlora.py --epochs 3
```

See [`qlora/SAGEMAKER_GUIDE.md`](qlora/SAGEMAKER_GUIDE.md) for the full walkthrough.
