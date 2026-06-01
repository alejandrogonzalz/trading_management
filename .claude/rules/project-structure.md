# Project Structure

## Where Things Live

| What | Where | Notes |
|------|-------|-------|
| ML models code | `langgraph/backtest/models/` | features.py, sklearn_models.py, lstm.py |
| Optimization | `langgraph/optimization/` | pipeline.py, searchers/, configs/ |
| Dataset | `langgraph/backtest/data/labeled/dataset.jsonl` | 56K samples, NEVER modify |
| Raw candles | `langgraph/backtest/data/candles/` | Per-symbol JSON, tracked by DVC |
| Backtest results | `langgraph/backtest/data/results/` | One JSON per experiment tag |
| Optimization results | `langgraph/optimization/results/` | {model}_optimization.json |
| Notebooks | `langgraph/Avance*.ipynb` | One per assignment deliverable |
| Agent graph | `langgraph/agent/` | LangGraph 3-node DAG |
| Backend API | `backend/app/` | FastAPI + Binance |
| Frontend | `frontend/` | React + Vite |

## Virtual Environment
- Path: `langgraph/.venv/` (Python 3.11)
- Activate: `cd langgraph && source .venv/bin/activate`
- Key deps: torch, scikit-learn, xgboost, pandas, numpy, matplotlib, ta-lib

## Running Scripts
Always prefix with `OMP_NUM_THREADS=1` when mixing sklearn/xgboost with PyTorch:
```bash
cd langgraph && OMP_NUM_THREADS=1 .venv/bin/python <script>
```
