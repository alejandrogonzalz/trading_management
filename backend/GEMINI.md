# Trading Agent Ecosystem

This workspace contains two distinct but integrated repositories for autonomous and quant-driven cryptocurrency trading.

## 1. Binance MCP (`/binance-mcp`) ✅ STABLE
A Python-based **Model Context Protocol (MCP)** server that connects the Gemini CLI directly to Binance. It is the core execution engine for low-level tasks.

### Core Features:
- **Market Data**: Direct access to order books, ticker prices, and historical OHLCV.
- **Account Management**: Real-time balance checking and trade history.
- **Lead Trading**: Initial support for Copy Trading and Futures Lead Orders.
- **Safety**: Built-in `recvWindow` safety and clock synchronization logic.

---

## 2. Trading Management Workstation (`/trading_management`) ✅ PRODUCTION READY
A full-stack **Quant Trading Platform** with a service-oriented backend and a React-based professional dashboard.

### Latest Progress (March 10, 2026):
- **MongoDB Migration**: Fully persistent document-based storage for trades, audit logs, and system metrics.
- **Truth-Based Reconciler**: Background watcher that audits specific Binance Order IDs every 30s. Handles `FILLED` (auto-archive) and `CANCELLED` (manual control) states.
- **Smart Trade Lifecycle**: 
  - **Isolated Virtual Positions**: Each manual trade is a separate entity with unique ID fingerprinting.
  - **Single-Leg Support**: Supports setups with ONLY Take Profit or ONLY Stop Loss.
  - **Fee-Aware Clipping**: Prevents "Insufficient Balance" errors by auto-deducting commissions from sell quantities.
- **Quantum Scanner & AI Ranking**:
  - **Multithreaded Turbo-Scan**: Fetches 140+ indicators across 7 timeframes in < 3 seconds.
  - **AI Setup Engine**: Uses **Qwen 2.5 14B** to rank top 20 coins and generate precise Entry/TP/SL targets.
  - **Market Guard**: Proactively filters out symbols in `BREAK` or `HALT` status.
- **Enhanced Dashboard UI**:
  - **Smart Trades View**: Visual progress bars showing SL -> Entry -> Price -> TP.
  - **Closed Trades History**: Precise P&L tracking including total dollar profit and exchange fees.
- **Audit System**: Permanent request/response logging for every Binance interaction.

### Tech Stack:
- **Backend**: FastAPI, MongoDB (Motor), TA-Lib, Python-Binance (Latest), Async Ollama (GPU).
- **Frontend**: React (Vite), TailwindCSS, Lucide-React, Lightweight Charts.
- **Infrastructure**: Docker Compose with persistent volumes.

---

## Mandatory Operational Protocols

### 🛠 Git Protocol (CRITICAL)
In this environment (Win32/PowerShell), the `&&` operator often fails.
*   **DO NOT** chain git commands with `&&`.
*   **ALWAYS** execute git actions sequentially in separate tool calls or use `;`.

### 🛡️ Smart Trade Safety
*   The system uses **Floor Rounding** for all quantities.
*   The system uses **ID-Strict Reconciliation**. Do not delete trades from MongoDB if Binance orders are active.

---

### Future Roadmap: The "Universal Terminal" & Lead Trading

1.  **Lead Trading Execution (HIGH PRIORITY)**: 
    *   Implement `lead_trading_service.py` to handle specialized USDS-M Futures orders.
    *   Integrate `binance-sdk-copy-trading` for monitoring and status checks.
    *   Create dedicated `lead_trades` MongoDB collection to ensure 100% isolation from Spot system.
2.  **Trailing Stops**: Implement dynamic trailing stop-loss logic for both Spot and Lead.
3.  **Smart Trade History Expansion**:
    *   Add "Start Date/Time" column to the history table.
    *   Include more granular metrics (ROI, Risk/Reward Ratio achieved).
    *   **Timestamp Accuracy**: Fix `closed_at` to use the actual Binance order fill time instead of the reconciler's detection time.
4.  **Terminal UX Improvements**:
    *   Allow direct manual entry for TP/SL percentage fields (completed).
    *   Add "Partial Exit" buttons (25%/50%/75%) to Active Positions.
5.  **Multi-Exchange Support**: Abstract the `binance_service` to allow other MCP-enabled exchanges in the future.
