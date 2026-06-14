# Project Structure

## Where Things Live

| What | Where | Notes |
|------|-------|-------|
| ML models code | `langgraph/backtest/models/` | features.py, sklearn_models.py, lstm.py |
| Optimization | `langgraph/optimization/` | pipeline.py, searchers/, configs/ |
| QLoRA fine-tuning | `langgraph/optimization/qlora/` | train_qlora.py, run_3configs.sh, sagemaker_setup.sh |
| QLoRA results | `langgraph/optimization/qlora/results/` | qlora_config*.json |
| Dataset | `langgraph/backtest/data/labeled/dataset.jsonl` | 56K samples, NEVER modify |
| Raw candles | `langgraph/backtest/data/candles/` | Per-symbol JSON, DVC-tracked |
| Trained models | `langgraph/backtest/data/models/` | LSTM (.pt), QLoRA adapters+GGUF, DVC-tracked |
| Backtest results | `langgraph/backtest/data/results/` | One JSON per experiment tag |
| Optimization results | `langgraph/optimization/results/` | {model}_optimization.json |
| Notebooks | `langgraph/Avance*.ipynb` | One per assignment deliverable |
| Agent graph | `langgraph/agent/` | LangGraph 3-node DAG |
| Backend API | `backend/app/` | FastAPI + Binance |
| Frontend | `frontend/` | React + Vite |
| Fine-tuning docs | `trading_management_docs/fine-tunning/` | FAQ, plan, local guide, briefing |

## Virtual Environments
- ML/backtest: `langgraph/.venv/` (Python 3.11) — torch, sklearn, xgboost
- QLoRA (local Windows): `langgraph/.venv-finetuning/` — unsloth, trl, bitsandbytes
- QLoRA (SageMaker): `langgraph/.venv/` (created by sagemaker_setup.sh)

## DVC (Data Version Control)
- Remote: `s3://trading-management-dvc/`
- Tracked files: dataset.jsonl, candles/*.json, models/qlora_config*/
- Pull: `cd langgraph && dvc pull`

## Running Scripts
Always prefix with `OMP_NUM_THREADS=1` when mixing sklearn/xgboost with PyTorch:
```bash
cd langgraph && OMP_NUM_THREADS=1 .venv/bin/python <script>
```
