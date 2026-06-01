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

## Current Status (2026-06-01)
- **Production**: Backend + Frontend + Ollama running via Docker Compose
- **LangGraph**: Agent running, 3-node graph tested
- **Backtest**: Refactored into clean subpackages (`ingestion/`, `models/`, `evaluation/`)
- **ML Results (Avance4)**: LSTM 81.5% test acc (winner), SVM 70.8%, XGBoost 66.6%
- **LSTM Best Params**: hidden=32, layers=3, seq_len=5, dropout=0.1, lr=0.001
- **Next**: Avance5 — ensemble models (stacking, voting, bagging-LSTM, blending)
- **Pending**: QLoRA fine-tuning, DVC, full LLM vs ML thesis comparison

---

## Steerings (Detailed Context Files)

@.claude/steering-backend.md
@.claude/steering-frontend.md
@.claude/steering-langgraph.md

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
