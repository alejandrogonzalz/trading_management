# Trading Management System (V2)

Advanced crypto trading scanner and management system with GPU-accelerated AI ranking.

## New Features (Turbo-Scan)

- **Multithreaded Scanner**: Fetches and computes data for 20+ pairs across 7 timeframes (5m to 1M) in ~3 seconds using nested `ThreadPoolExecutor`.
- **Opportunity Ranking**: Automatically discovers the "Top 20" most interesting USDC pairs based on a weighted multi-factor score:
  - 30% Volume (>1M USDC)
  - 25% Volatility (ATR)
  - 20% Momentum (RSI)
  - 15% Trend Strength (ADX)
  - 10% Recent Move
- **GPU AI Ranking**: Integrates with local Ollama (NVIDIA GPU) to rank setups and provide technical reasoning.
- **Enhanced Heatmap**: 3-EMA (20/50/200) trend stack with Price Confirmation and ADX Strength filters.
- **Interactive UI**: Fully sortable and filterable quant table with detailed hover tooltips for every indicator.

## Tech Stack

- **Backend**: FastAPI, TA-Lib (C-Library), Pydantic V2, APScheduler.
- **Frontend**: React (Vite), TailwindCSS, Lucide-React.
- **AI**: Ollama (Qwen 2.5 7B) with NVIDIA GPU Passthrough.
- **Infrastructure**: Docker Compose with GPU reservation.

## Installation

1. Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed.
2. Configure `.env` in `backend/` with `BINANCE_API_KEY` and `BINANCE_API_SECRET`.
3. Launch:
   ```bash
   docker-compose up -d --build
   ```

## Development History
Successfully restructured from a monolithic backend to a service-oriented architecture (Market, Indicator, Scoring, Scanner, and LLM services).
