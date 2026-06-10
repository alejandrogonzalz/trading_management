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
│   └── qlora.yaml
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

## QLoRA Fine-Tuning Setup (Windows — RTX 5070 Ti)

Steps required to run `train_qlora.py` on a fresh Windows machine.

### 1. Install AWS CLI and configure credentials

The labeled dataset is stored in a DVC S3 remote (`s3://trading-management-dvc/dvc`).
AWS CLI is needed to pull it.

```powershell
# Download and install AWS CLI v2
Invoke-WebRequest -Uri "https://awscli.amazonaws.com/AWSCLIV2.msi" -OutFile "$env:TEMP\AWSCLIV2.msi"
Start-Process msiexec.exe -ArgumentList "/i $env:TEMP\AWSCLIV2.msi /quiet /norestart" -Wait
# Restart terminal to pick up the new PATH, then:
aws configure   # enter Access Key, Secret Key, region us-east-1, output json
aws sts get-caller-identity   # confirm auth
```

### 2. Install DVC + dvc-s3 in the finetuning venv

```powershell
.venv-finetuning\Scripts\pip install dvc dvc-s3
```

### 3. Pull the dataset

```powershell
# From the repo root:
.venv-finetuning\Scripts\dvc pull langgraph/backtest/data/labeled/dataset.jsonl
# Expected: "1 file fetched and 1 file added"
# Verify:
python -c "print(sum(1 for _ in open('langgraph/backtest/data/labeled/dataset.jsonl')))"
# → 56161
```

### 4. Install CUDA-enabled PyTorch

Unsloth installs `torch` during its own install, but on Windows it may resolve the
`+cpu` wheel instead of a CUDA build. Always verify and fix before training:

```powershell
# Check — if output contains "+cpu", reinstall:
.venv-finetuning\Scripts\python -c "import torch; print(torch.__version__, torch.cuda.is_available())"

# Install CUDA 12.8 build (works with driver CUDA 13.x via backward compat):
.venv-finetuning\Scripts\pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128 --upgrade

# Verify CUDA is now available:
.venv-finetuning\Scripts\python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
# → True  NVIDIA GeForce RTX 5070 Ti
```

> **Why this happens**: Unsloth pulls the correct CUDA wheel on Linux/Colab, but on
> Windows the pip resolver sometimes falls back to the CPU wheel when no explicit
> `--index-url` is given. The `cu128` index works with any driver that supports
> CUDA ≥ 12.8 (driver ≥ 525.x), including the 591.86 driver shipping with
> CUDA 13.1.

### 5. Run fine-tuning

```powershell
Set-Location langgraph
.venv-finetuning\Scripts\python optimization/train_qlora.py --lr 0.00002 --rank 16 --epochs 3 --batch-size 4
```

Expected VRAM usage: ~10-13 GB (out of 16 GB), training ~4-8 h for 3 epochs.
