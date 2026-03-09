# Trading Management (Smart Trade)

A Dockerized 3Commas-style "Smart Trade" management system. It separates your trades into **USDC** pairs to avoid interference with other bots and provides an interface for placing entries with automated Take Profit (Limit) and Stop Loss (Stop Limit) orders.

## Features

- **USDC Focus**: Only monitors and trades USDC pairs.
- **Smart Trade Interface**: Enter trades with pre-defined TP/SL targets.
- **Real-time Dashboard**: Monitor balances and open orders.
- **Dockerized**: Start both backend and frontend with one command.

## Tech Stack

- **Frontend**: React, TypeScript, Tailwind CSS, Lightweight Charts.
- **Backend**: Python (FastAPI), `python-binance`.
- **Infrastucture**: Docker Compose.

## Getting Started

1. **Configure Environment**:
   Ensure you have a `.env` file in the root of this directory with:
   ```env
   BINANCE_API_KEY=your_key
   BINANCE_API_SECRET=your_secret
   ```

2. **Run with Docker Compose**:
   ```bash
   docker-compose up --build
   ```

3. **Access the UI**:
   Open [http://localhost:5173](http://localhost:5173) in your browser.

## Architecture

- **Backend (Port 8001)**: Serves as a secure proxy to Binance and handles Smart Trade logic using OCO orders.
- **Frontend (Port 5173)**: React dashboard for visualization and trade execution.

## License

MIT
