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

### Future Roadmap: Lead Trading & Universal Terminal

#### 1. Backend: Futures Engine (COMPLETED ✅)
- [x] Implement `futures_service.py` using specialized Binance SDKs.
- [x] Create modular routes in `app/routes/lead.py`.
- [x] Support for Leverage, Lead Orders, and Position Monitoring.

#### 2. Frontend: Global State & Navigation
- [x] **Dynamic Environment Switching**: Toggle between "Spot" and "Lead" modes in the header.
- [x] **Context-Aware Search**: Automatically filter the symbol search bar based on the active mode (Spot symbols vs. Lead Whitelist).
- [ ] **Unified Mode Sync**: Ensure that clicking a symbol in the scanner automatically respects the current trading mode.

#### 3. Frontend: Smart Terminal & Deep Analysis
- [x] **Leverage Management**: Add a high-precision slider (1x - 50x) to the terminal for Futures setups.
- [x] **Short/Long Support**: Fully implement the dual-side logic for Lead Positions.
- [ ] **Deep Analysis "Setup Futures"**: Add a dedicated button in the AI Analysis popup to generate leverage-optimized setups.
- [x] **Pre-Trade Risk**: Display estimated Liquidation Price and Margin Required before clicking "Execute."

#### 4. Frontend: Monitoring & Wallet
- [ ] **Lead Positions Table**: New view dedicated to live Futures tracking (Size, Entry, Mark Price, Liq Price, ROE%).
- [ ] **Mega-Tooltip 2.0**: Expand the wallet hover to show a detailed "Futures Wallet" section alongside "Spot Wallet."
- [ ] **Margin Health**: Visual "Margin Ratio" indicator to prevent liquidations.

#### 5. Frontend: Scanner & History
- [ ] **Scanner Whitelist Toggle**: Filter the scanner results to only show symbols eligible for Lead Trading.
- [ ] **Advanced History**: Add "Mode" (Spot/Lead) and "Leverage" columns to the Closed Trades tab.
- [ ] **Performance Analytics**: Include Futures-specific metrics (Funding Fees, ROE) in the header stats panel.

