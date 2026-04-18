# Trading Agent Ecosystem

A full-stack crypto trading platform with GPU-accelerated AI ranking, autonomous workflows, and professional dashboard.

## Project Overview

The Trading Management System is a production-ready quant trading platform featuring:
- **Multithreaded Turbo-Scanner**: Analyzes 20+ pairs across 7 timeframes in ~3 seconds
- **AI Ranking & Setup Generation**: Local Qwen 2.5 14B model for technical analysis and trade setup generation
- **Smart Trade Lifecycle**: Set-and-forget OCO orders with fee-aware clipping and automatic reconciliation
- **Lead (Futures) Trading**: Leveraged positions with panic-sell and atomic rollback safety
- **Professional Dashboard**: React-based UI with real-time charts, scanner table, and portfolio monitoring

## Tech Stack

### Backend (FastAPI)
- **Framework**: FastAPI with Pydantic V2 validation
- **Database**: SQLite with SQLAlchemy ORM (WAL mode for concurrency)
- **Market Data**: Python-Binance SDK, TA-Lib for technical indicators
- **AI Integration**: Ollama with Qwen 2.5 14B (GPU-accelerated)
- **Scheduling**: APScheduler for background tasks (reconciliation every 30s)
- **Audit Logging**: Comprehensive request/response logging to SQLite

### Frontend (React)
- **Framework**: React 19 with TypeScript
- **Build Tool**: Vite for ultra-fast development
- **UI Components**: TailwindCSS, Lucide-React icons
- **Charts**: Lightweight Charts by TradingView
- **State Management**: React hooks with context API

### Infrastructure
- **Containerization**: Docker Compose with GPU passthrough
- **GPU Support**: NVIDIA Container Toolkit for Ollama acceleration
- **Networking**: Custom bridge network for service communication
- **Persistent Storage**: Volume mounts for SQLite database and Ollama models

## Installation & Setup

### Prerequisites
1. **NVIDIA GPU**: Install [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
2. **Binance API Keys**: Create API keys with trading permissions
3. **Docker & Docker Compose**: Latest versions installed

### Configuration
1. **Backend Environment**: Create/update `backend/.env`:
   ```
   BINANCE_API_KEY=your_api_key_here
   BINANCE_API_SECRET=your_api_secret_here
   BINANCE_BASE_URL=https://api.binance.com
   
   # Optional: Lead Trading credentials
   BINANCE_COPY_TRADING_KEY=your_lead_key_here
   BINANCE_COPY_TRADING_SECRET=your_lead_secret_here
   ```

2. **Launch Services**:
   ```bash
   docker-compose up -d --build
   ```

### Service Ports
- **Backend API**: http://localhost:8001
- **Frontend Dashboard**: http://localhost:5173  
- **LangGraph Agent**: http://localhost:2024
- **Ollama LLM**: http://localhost:11434

## Core Features

### 1. Quantum Scanner & AI Ranking
- **Turbo-Scan Engine**: Multithreaded analysis using nested `ThreadPoolExecutor`
- **Multi-Factor Scoring**: Weighted ranking based on:
  - 30% Volume (>1M USDC)
  - 25% Volatility (ATR)
  - 20% Momentum (RSI)
  - 15% Trend Strength (ADX)
  - 10% Recent Move
- **AI Filtering**: Local LLM identifies top 3 setups with highest technical confluence
- **Deep Analysis**: Per-coin timeframe analysis with precise Entry/TP/SL targets

### 2. Smart Trade Lifecycle
- **Set-and-Forget Architecture**: Capital protected even if computer is offline
- **Fee-Aware Clipping**: Auto-deducts commissions from sell quantities to prevent "Insufficient Balance" errors
- **OCO Orders**: One-Cancels-the-Other protection with fallback to single Limit/Stop orders
- **Floor Rounding**: Consistent quantity formatting across all operations

### 3. Lead (Futures) Trading
- **Leverage Management**: 1x-50x adjustable leverage with real-time liquidation price calculation
- **Atomic Execution**: "Verify-then-Proceed" architecture with position verification
- **Panic Sell**: One-click liquidation of position and cancellation of all open orders
- **Reconciliation**: Background capture of exit fees and accurate P&L calculation

### 4. Autonomous Reconciliation
- **30-Second Sync**: Background job queries Binance for every stored Order ID
- **Fill Detection**: Checks Binance Trade History for filled TP/SL orders
- **Auto-Closure**: Captures exit price/fees and marks trades as `CLOSED`
- **Manual Control Detection**: Switches to `MANUAL_CONTROL` mode if orders cancelled externally

## Database Schema

### SQLite Database (`trading.db`)

#### Table: `trades` (Spot Trading)
```sql
id TEXT PRIMARY KEY,           -- Unique trade identifier
symbol TEXT,                   -- Trading pair (e.g., "BTCUSDT")
side TEXT,                     -- "BUY" or "SELL"
entry_price REAL,              -- Weighted average entry price
quantity REAL,                 -- Floor-rounded quantity
tp REAL,                       -- Take profit price
sl REAL,                       -- Stop loss price
status TEXT,                   -- "ACTIVE", "CLOSED", "MANUAL_CONTROL"
orders JSON,                   -- Array of Binance order IDs
entry_fees REAL,               -- Entry commission amount
fee_asset TEXT,                -- Commission asset (e.g., "BNB")
exit_price REAL,               -- Exit fill price
exit_fees REAL,                -- Exit commission amount
exit_fee_asset TEXT,           -- Exit commission asset
close_time REAL,               -- Unix timestamp of closure
timestamp REAL,                -- Trade creation timestamp
close_reason TEXT,             -- "TP_FILLED", "SL_FILLED", "MANUAL"
error_msg TEXT                 -- Any error during trade lifecycle
```

#### Table: `lead_trades` (Futures Trading)
```sql
id TEXT PRIMARY KEY,           -- Unique futures trade identifier
symbol TEXT,                   -- Trading pair
side TEXT,                     -- "LONG" or "SHORT"
entry_price REAL,              -- Entry price
quantity REAL,                 -- Position size
leverage INTEGER,              -- Leverage multiplier (1-50)
tp REAL,                       -- Take profit price
sl REAL,                       -- Stop loss price
status TEXT,                   -- "ACTIVE", "CLOSED"
protection_orders JSON,        -- Array of SL/TP order IDs
entry_fees REAL,               -- Entry commission
entry_fee_asset TEXT,          -- Commission asset (default "USDT")
exit_price REAL,               -- Exit price
exit_fees REAL,                -- Exit commission
exit_fee_asset TEXT,           -- Exit commission asset
close_time REAL,               -- Closure timestamp
close_reason TEXT,             -- Closure reason
timestamp REAL,                -- Creation timestamp
error TEXT                     -- Error details if any
```

#### Table: `audit_log`
```sql
id INTEGER PRIMARY KEY AUTOINCREMENT,
timestamp_str TEXT,            -- Human-readable timestamp
timestamp REAL,                -- Unix timestamp
method TEXT,                   -- HTTP method
endpoint TEXT,                 -- API endpoint
request JSON,                  -- Request payload
response JSON,                 -- Response payload
status INTEGER                 -- HTTP status code
```

#### Table: `endpoint_audit`
```sql
id INTEGER PRIMARY KEY AUTOINCREMENT,
timestamp TEXT,                -- Request timestamp
unix_time REAL,                -- Unix timestamp
method TEXT,                   -- HTTP method
url TEXT,                      -- Full URL
status_code INTEGER,           -- Response status
process_time_ms REAL,          -- Request processing time
request_body JSON              -- Request body (if POST/PUT)
```

## Application Workflows

### 1. Smart Trade Execution
```
1. Entry Phase → MARKET order execution with fills array parsing
2. Calculation → Floor rounding + fee-aware quantity clipping
3. Protection → OCO order placement (TP + SL)
4. Database → Trade record with Master ID and Binance Order IDs
5. Monitoring → UI progress bar with real-time P&L calculation
```

### 2. Scanner → AI Ranking Pipeline
```
1. Discovery → Fetch 24hr stats, filter TRADING symbols > $1M volume
2. Scoring → Analyze top 20 across 7 timeframes, assign Quant Score (0-10)
3. AI Filtering → Send table to Qwen 2.5 for top 3 setup identification
4. Deep Analysis → Per-coin timeframe analysis with precise trade setup
```

### 3. Reconciliation System
```
Every 30 seconds:
1. Query Binance for each stored Order ID
2. Detect FILLED status in Binance Trade History
3. Capture exit price/fees, mark trade as CLOSED
4. Handle CANCELLED orders → MANUAL_CONTROL mode
```

### 4. Lead Trading Engine
```
1. Position Verification → Poll Position Risk endpoint
2. Order Verification → Confirm Algo orders (SL/TP) on order book
3. Atomic Rollback → Cancel All + Market Close on failure
4. Fee Capture → Record exit fees from Binance trade history
```

## Mandatory Operational Protocols

### 🛠 Git Protocol (Windows/PowerShell)
In this environment, the `&&` operator often fails:
- **DO NOT** chain git commands with `&&`
- **ALWAYS** execute git actions sequentially in separate tool calls or use `;`

### 🛡️ Smart Trade Safety
- **Floor Rounding**: Applied to all quantities for consistency
- **ID-Strict Reconciliation**: Never delete trades from database while Binance orders are active.
- **Downtime Accuracy (KNOWN LIMITATION)**: If the app is offline for an extended period, the Lead/Futures reconciler may use the *current* ticker price as the `exit_price` if it can't find a matching protection order fill. This will be improved in a future update to scan historical fills.
- **Fee-Aware Clipping**: Commissions auto-deducted from sell quantities
- **Time Synchronization**: `recvWindow=60000` with automatic clock sync

### 🔧 Development Guidelines
- **SQLite Concurrency**: WAL mode enabled for better multi-process access
- **Audit Trail**: All API interactions logged to `audit_log` table
- **Error Handling**: Comprehensive error capture with rollback mechanisms
- **Testing**: Pytest with coverage for critical trading functions

## Future Roadmap

### Backend Enhancements
- [ ] **Trailing Stop**: Dynamic trailing for Lead positions
- [ ] **Multi-Exchange Support**: Abstract futures_service for Bybit/OKX integration
- [ ] **Advanced Risk Management**: Position sizing algorithms, correlation analysis
- [ ] **WebSocket Streaming**: Real-time order book and ticker data

### Frontend Improvements
- [ ] **Mobile Optimization**: Responsive design for smaller screens
- [ ] **Lead Positions Table**: Dedicated Futures tracking view
- [ ] **Mega-Tooltip 2.0**: Expanded wallet hover with Futures/Spot breakdown
- [ ] **Advanced Analytics**: Performance metrics, win rate analysis, Sharpe ratio
- [ ] **Scanner Whitelist Toggle**: Filter for Lead-eligible symbols only

### AI & Automation
- [ ] **Autonomous Trading**: LangGraph agents for fully automated decision making
- [ ] **Multi-Model Support**: Integration with additional local LLMs (Llama, Mistral)
- [ ] **Backtesting Engine**: Historical strategy simulation with walk-forward analysis
- [ ] **Pattern Recognition**: Machine learning for chart pattern detection

## Troubleshooting

### Common Issues

1. **"Insufficient Balance" errors**
   - Verify fee-aware clipping is working
   - Check for correct quantity floor rounding
   - Ensure sufficient USDT/BNB for commission fees

2. **API Timeout/Rejection**
   - Run `binance_service.sync_time()` to synchronize clocks
   - Verify `recvWindow=60000` is being used
   - Check Binance API key permissions

3. **GPU/Ollama Issues**
   - Verify NVIDIA Container Toolkit installation
   - Check GPU passthrough in docker-compose.yml
   - Monitor Ollama model loading via logs

4. **Database Connection Issues**
   - Ensure SQLite file has correct permissions
   - Verify WAL mode is enabled (`PRAGMA journal_mode=WAL`)
   - Check for database file corruption

### Monitoring & Logs
- **Backend Logs**: `docker-compose logs backend`
- **Frontend Logs**: `docker-compose logs frontend`
- **Database Inspection**: Use SQLite browser or `sqlite3 trading.db`
- **Audit Trail**: Query `audit_log` table for API interaction history

## License & Contributions

MIT Licensed. Contributions welcome via pull requests with comprehensive testing.

---

*Last Updated: April 2026*  
*Current Version: Production Ready*  
*Database: SQLite (SQLAlchemy)*  
*AI Model: Qwen 2.5 14B (GPU-accelerated)*