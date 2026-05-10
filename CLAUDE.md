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

## Current Status (2026-05-09)
- **Production**: Backend + Frontend + Ollama running via Docker Compose
- **LangGraph**: Agent running, 3-node graph tested
- **ML Results**: LSTM 84.29% acc (winner), XGB 76.79%, RF 74.89%
- **Pending**: LSTM test-set eval, serialization, QLoRA fine-tuning, DVC

---

## Steerings (Detailed Context Files)

@.claude/steering-backend.md
@.claude/steering-frontend.md
@.claude/steering-langgraph.md
