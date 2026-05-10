# Backend — FastAPI + Binance + SQLite

## Stack
| Layer | Tech |
|-------|------|
| Framework | FastAPI + Pydantic V2 |
| Database | SQLite + SQLAlchemy ORM (WAL mode) |
| Market Data | python-binance SDK + TA-Lib |
| AI (local) | Ollama + Qwen 2.5 14B (GPU via Docker) |
| Scheduling | APScheduler — reconciliation every 30s |
| Audit | audit_log + endpoint_audit tables |

## Directory Structure
```
backend/app/
├── core/
│   ├── config.py       # env vars, settings
│   └── middleware.py   # request logging, CORS
├── db/
│   ├── database.py     # SQLAlchemy engine, session, WAL setup
│   └── models.py       # ORM models: Trade, LeadTrade, AuditLog
├── routes/
│   ├── spot.py         # /spot/* — spot trade execution
│   ├── lead.py         # /lead/* — futures/leveraged trades
│   └── market.py       # /market/* — scanner, ranking, analysis
├── services/
│   ├── binance_service.py    # Spot API: orders, fills, sync_time()
│   ├── futures_service.py    # Futures API: leverage, positions, TP/SL
│   ├── scanner_service.py    # Turbo-scan: multithreaded, 20+ pairs × 7 TFs
│   ├── scoring_service.py    # Quant Score: weighted 5-factor scoring
│   ├── llm_service.py        # Ollama calls for AI ranking + deep analysis
│   ├── market_service.py     # 24hr stats, ticker, klines
│   ├── indicator_service.py  # TA-Lib wrapper for live analysis
│   ├── trade_tracker.py      # Trade CRUD + status management
│   ├── audit_service.py      # audit_log writes
│   └── market_utils.py       # Floor rounding, fee clipping, formatting
├── models.py           # Pydantic request/response schemas
└── main.py             # App entrypoint, lifespan, router registration
```

## Database Schema

### `trades` (Spot)
```sql
id TEXT PK          -- Master trade ID
symbol TEXT         -- e.g. "BTCUSDT"
side TEXT           -- "BUY" | "SELL"
entry_price REAL    -- Weighted avg fill
quantity REAL       -- Floor-rounded
tp REAL             -- Take profit
sl REAL             -- Stop loss
status TEXT         -- "ACTIVE" | "CLOSED" | "MANUAL_CONTROL"
orders JSON         -- [binance_order_id, ...]
entry_fees REAL
fee_asset TEXT      -- e.g. "BNB"
exit_price REAL
exit_fees REAL
exit_fee_asset TEXT
close_time REAL     -- Unix timestamp
timestamp REAL      -- Creation timestamp
close_reason TEXT   -- "TP_FILLED" | "SL_FILLED" | "MANUAL"
error_msg TEXT
```

### `lead_trades` (Futures)
```sql
id TEXT PK
symbol TEXT
side TEXT           -- "LONG" | "SHORT"
entry_price REAL
quantity REAL
leverage INTEGER    -- 1-50
tp REAL
sl REAL
status TEXT         -- "ACTIVE" | "CLOSED"
protection_orders JSON   -- [sl_order_id, tp_order_id]
entry_fees REAL
entry_fee_asset TEXT     -- default "USDT"
exit_price REAL
exit_fees REAL
exit_fee_asset TEXT
close_time REAL
close_reason TEXT
timestamp REAL
error TEXT
```

### `audit_log` / `endpoint_audit`
Full HTTP request/response logging for all API calls.

## Core Trade Workflows

### Spot Trade Execution
1. MARKET buy → parse fills array → weighted avg entry_price
2. Fee-aware clipping: `sell_qty = floor(qty - fee_qty)`
3. OCO order (TP limit + SL stop-limit)
4. Save trade with master ID + Binance order IDs
5. APScheduler reconciles every 30s

### Reconciliation (every 30s)
1. For each ACTIVE trade: query Binance order status
2. If FILLED → capture exit price + fees → mark CLOSED
3. If CANCELLED → mark MANUAL_CONTROL
4. Lead trades: check position risk + protection orders

### Lead Trade Execution
1. Set leverage via Futures API
2. MARKET position open
3. Verify position exists (poll Position Risk)
4. Place TP + SL algo orders
5. Verify orders on order book
6. Atomic rollback on any failure (cancel all + market close)

## Scanner & AI Ranking

### Quant Score (0-10)
| Weight | Factor |
|--------|--------|
| 30% | Volume (>$1M USDC) |
| 25% | Volatility (ATR) |
| 20% | Momentum (RSI) |
| 15% | Trend Strength (ADX) |
| 10% | Recent Price Move |

### Scanner Flow
1. Fetch 24hr stats → filter TRADING symbols > $1M vol
2. Multithreaded: `ThreadPoolExecutor` (nested), 20 pairs × 7 TFs
3. Score top candidates → send to Qwen 2.5 via Ollama
4. LLM identifies top 3 setups + deep analysis per coin

## Key Binance Patterns
```python
# Always use recvWindow=60000
# Sync clocks on Timestamp errors
binance_service.sync_time()

# Floor rounding (critical)
from app.utils.market_utils import floor_round
qty = floor_round(raw_qty, step_size)
```

## Troubleshooting
| Symptom | Fix |
|---------|-----|
| "Insufficient Balance" | Check fee-aware clipping + floor rounding |
| API Timestamp error | Call `sync_time()`, verify `recvWindow=60000` |
| GPU/Ollama slow | Check `docker-compose logs ollama`, NVIDIA Toolkit |
| DB locked | WAL mode handles concurrent reads; check for stale write lock |
