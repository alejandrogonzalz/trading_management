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

### Latest Progress (March 12, 2026):
- **Full UI/UX Overhaul**:
  - **Mega-Tooltip Wallet**: Hoverable portfolio breakdown with live Binance exchange rates and total USDT valuation.
  - **Performance Stats Panel**: Real-time calculation of Win Rate, Net P&L, and Avg Profit in the Smart Trades view.
  - **Clickable Navigation**: One-click jump to terminal from any symbol in Scanner, Orders, or History views.
  - **Notification System**: Replaced browser alerts with sleek, non-blocking integrated Toasts.
- **Architectural Stabilization**:
  - **Modular Views**: Refactored `App.tsx` into isolated, scalable components (`BottomPanel`, `SmartTerminalView`, `WalletBalances`).
  - **Grid-Based Layout**: Transitioned to CSS Grid for a rock-solid, responsive terminal that prevents panel vanishing.
  - **Sticky Components**: Implemented sticky headers for all data tables ensuring context is never lost during scrolling.
- **Precision Trade Lifecycle**:
  - **Accurate Fee Capture**: Backend now records `exit_fee_asset` and sums exact commissions at the moment of fill.
  - **Dollar-Value Normalization**: Frontend intelligently converts asset-based fees (e.g., SOL/BTC) to USDT for precise P&L history.
  - **Binance Fill-Time Accuracy**: Reconciler and Panic Sell now use Binance `transactTime` instead of system detection time.
- **Database Maintenance**:
  - **Historical Repair**: Ran a diagnostic script to fix unit errors and missing assets in older trade records.
  - **Utility Organization**: Created a dedicated `/scripts` folder for database maintenance and emergency tools.

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

### Future Roadmap: Lead Trading & Universal Terminal (COMPLETED ✅)

#### 1. Backend: Futures Engine (COMPLETED ✅)
- [x] **Bulletproof Smart Trade**: Implemented "Verify-then-Proceed" architecture for SL/TP.
  - Position verification (polling).
  - Standard Order Engine (for reliability).
  - Atomic Rollback (Cancel All + Market Close).
- [x] **Panic Sell**: One-click liquidation of position and cancellation of all open orders.
- [x] **Reconciliation**: Background job to capture exit fees and accurate P&L.
- [x] **Algo Engine Integration**: Correctly handling `STOP_MARKET` and `TAKE_PROFIT_MARKET` via Algo endpoints.

#### 2. Frontend: Universal Terminal (COMPLETED ✅)
- [x] **Smart Lead Trades**: Dedicated history view with Fees, Leverage, and Net P&L.
- [x] **Global History**: Bottom panel now aggregates raw Binance history + Smart Trade records.
- [x] **Active Filtering**: Cleaned up "Active Orders" to hide filled/canceled setups.
- [x] **Fee Accounting**: Visualization of exit fees in trade history.

#### 3. Next Steps (Optimization)
- [ ] **Trailing Stop**: Implement dynamic trailing for Lead positions.
- [ ] **Multi-Exchange**: Abstract `futures_service` for Bybit/OKX.
- [ ] **Mobile View**: Optimize `ActivePositionsView` for smaller screens.
