# Trading Agent Ecosystem — Project Hub

Full-stack crypto trading platform with GPU-accelerated AI, LangGraph agents, backtesting framework, and ML optimization pipeline.

## Quick Reference

| Component | Path | Port |
|-----------|------|------|
| FastAPI Backend | `backend/` | 8001 |
| React Frontend | `frontend/` | 5173 |
| LangGraph Agent | `langgraph/` | 2024 |
| Ollama LLM | Docker | 11434 |
| SQLite DB | `backend/trading.db` | — |

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│  React Frontend (port 5173)                         │
│  Vite + TypeScript + TailwindCSS + LightweightCharts│
└───────────────────┬─────────────────────────────────┘
                    │ REST
┌───────────────────▼─────────────────────────────────┐
│  FastAPI Backend (port 8001)                        │
│  Scanner → Scoring → Binance → SQLite               │
│  APScheduler reconciliation every 30s               │
└───────────┬───────────────────┬─────────────────────┘
            │                   │
    ┌───────▼──────┐    ┌───────▼──────────┐
    │ Ollama LLM   │    │ LangGraph Agent  │
    │ Qwen 2.5 14B │    │ (port 2024)      │
    │ (GPU Docker) │    │ 3-node DAG       │
    └──────────────┘    └──────────────────┘
                                │
                    ┌───────────▼──────────┐
                    │ Backtest Framework   │
                    │ 56K samples, 12 syms │
                    │ XGB/RF/LSTM models   │
                    └──────────────────────┘
```

## Domain Context
- **Spot Trading**: BUY/SELL with OCO protection (TP + SL), fee-aware clipping, auto-reconciliation
- **Lead/Futures Trading**: Leveraged LONG/SHORT (1-50x), panic-sell, atomic rollback
- **Scanner**: Multithreaded analysis of 20+ pairs × 7 timeframes → Quant Score (0-10)
- **AI Ranking**: Qwen 2.5 14B via Ollama identifies top 3 setups + deep analysis
- **LangGraph Agent**: 3-node DAG (generator → evaluator → optimizer) for structured setup generation
- **Backtest**: Historical evaluation of LLM vs ML predictions (56,161 labeled samples)
- **Research**: Master's thesis comparing zero-shot LLM, fine-tuned LLM, XGB, RF, LSTM

## Current Status (2026-06-13)
- **Production**: Backend + Frontend + Ollama running via Docker Compose
- **LangGraph**: Agent running, 3-node graph tested
- **Backtest**: Refactored into clean subpackages (`ingestion/`, `models/`, `evaluation/`)
- **ML Results (Avance5)**: Bagging-LSTM 83.37% test (winner), LSTM 81.5%, Blending 81.89%, XGBoost 66.6% — ⚠️ re-measure on the fixed split (Tarea 7)
- **Bagging-LSTM Best**: 5 bags, hidden=32, layers=3, seq=5, AUC-ROC=0.9157
- **Leakage fix (2026-06-13)**: `_temporal_split` now a strict temporal holdout (global timestamp sort + embargo); prior per-symbol split leaked → all model numbers above need re-measuring. See `docs/AUDITORIA_QLORA_LEAKAGE_OVERFITTING.md` + `docs/GUIA_IMPLEMENTACION_FIX_QLORA.md`
- **QLoRA Fine-tuning**: ~~config 1 92%~~ INVALID (leakage, archived). Re-train ONE model on the fix (`run_cloud.sh`); diagnostics now emit train/val/test gap + baseline + loss curve
- **DVC**: Initialized, dataset + candles + models tracked in S3 (`s3://trading-management-dvc/`)
- **Next**: `dvc pull`; re-train QLoRA + ML/ensembles on fixed split; zero-shot backtest; McNemar/t-test comparisons
- **Pending**: LangGraph integration of fine-tuned model, ensemble LSTM+LLM, thesis presentation

---

## Steerings (Detailed Context Files)

@.claude/steering-backend.md
@.claude/steering-frontend.md
@.claude/steering-langgraph.md
@.claude/steering-qlora.md

## Custom Commands

- `/run-backtest <model> [tuned]` — Run MLBacktestRunner with trade simulation
- `/train-model <model>` — Launch hyperparameter optimization
- `/compare <tag1> <tag2>` — Compare two backtest result files

## ML Quick Reference

```bash
# Activate env
cd langgraph && source .venv/bin/activate

# Train a model
OMP_NUM_THREADS=1 python optimization/optimize.py --model lstm --config optimization/configs/lstm.yaml

# Run backtest
python -m cli run-backtest --dataset backtest/data/labeled/dataset.jsonl --provider mock

# Run tests
python -m pytest tests/ -v
```
