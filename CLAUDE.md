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

## Current Status (2026-06-16)
- **Production**: Backend + Frontend + Ollama running via Docker Compose
- **LangGraph**: Agent running, 3-node graph tested
- **All measurements complete** — all models evaluated on the same strict temporal test split

| Model | Test Acc | Win Rate | Profit Factor |
|-------|----------|----------|---------------|
| QLoRA cloud (lr=2e-5, rank=16) | **88.03%** | 61.67% | 12.92 |
| QLoRA config3 (lr=1e-5, rank=32) | **87.87%** | 60.30% | 11.96 |
| Random Forest v2 | 64.24% | 56.40% | 2.23 |
| XGBoost v2 | 63.60% | 57.47% | 2.34 |
| Blending ensemble v2 | 63.59% | — | — |
| Zero-shot Qwen 7B | 58.49% | 28.42% | 1.35 |
| LSTM v2 | 51.51% | 46.74% | 1.49 |

- **McNemar test** (QLoRA vs zero-shot): chi²=557, p≈0 — statistically significant
- **Leakage fix**: `_temporal_split` is a strict temporal holdout (global sort + embargo). Pre-fix numbers (Avance 4–5) are invalid and archived.
- **DVC**: Dataset, candles, QLoRA models tracked in S3 (`s3://trading-management-dvc/`)
- **Analysis notebook**: `langgraph/optimization-results.ipynb` — Parts 1–6 complete (loss curves, accuracy bar, McNemar, trade metrics, equity curves)
- **Remaining**: `Avance6.ipynb` (thesis deliverable), LangGraph integration (GGUF → Ollama → agent), thesis write-up + defense (~Jun 26)

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
