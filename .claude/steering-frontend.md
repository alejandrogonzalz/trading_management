# Frontend — React 19 + TypeScript + Vite

## Stack
| Layer | Tech |
|-------|------|
| Framework | React 19 + TypeScript |
| Build | Vite (ultra-fast HMR) |
| Styling | TailwindCSS |
| Icons | Lucide-React |
| Charts | Lightweight Charts (TradingView) |
| State | React hooks + Context API |
| Container | Docker (port 5173) |

## Key Features
- **Scanner Table**: Real-time Quant Score rankings with color-coded indicators
- **Portfolio Monitor**: Active trades with P&L progress bars
- **TradingView Charts**: Candlestick + indicator overlays via Lightweight Charts
- **Wallet Hover**: Balance breakdown (Spot/Futures) as mega-tooltip
- **Lead Positions**: Futures position monitoring with liquidation price display

## API Integration
All calls go to backend at `http://localhost:8001`.
- `/market/scan` — Trigger turbo-scan + AI ranking
- `/spot/trade` — Place spot trade
- `/lead/trade` — Place futures trade
- `/lead/panic-sell/{id}` — Emergency position close
- `/market/portfolio` — Current trades + balances

## Pending Items (Roadmap)
- Mobile-responsive layout
- Dedicated Lead Positions table view
- Mega-Tooltip 2.0 — Futures/Spot wallet breakdown
- Advanced analytics: win rate, Sharpe ratio display
- Scanner whitelist toggle for Lead-eligible symbols only

## Dev Workflow
```powershell
# Inside frontend/
npm install
npm run dev       # Dev server (port 5173)
npm run build     # Production build
npm run lint      # ESLint
```
Or via Docker: `docker-compose up frontend`
