# Application Workflows & Architecture

This guide describes the end-to-end workflows that power the Trading Workstation.

## 1. The Smart Trade Lifecycle (Set-and-Forget) ✅

Designed to ensure your capital is protected even if your computer is offline.

1.  **Entry Phase**: System executes a `MARKET` order. It immediately parses the `fills` array from the response to calculate the exact **Weighted Average Entry Price** and **Entry Commissions**.
2.  **Calculation Phase**: System uses **Floor Rounding** to format the quantity. It applies **Fee-Aware Clipping** (subtracting commissions from the total) to ensure the subsequent Sell order doesn't exceed the wallet balance.
3.  **Protection Phase**: System places an **OCO (One-Cancels-the-Other)** order. If the market doesn't support OCO for that coin, it falls back to a single **Limit Sell** or **Stop Loss**.
4.  **Database Phase**: The Master ID, Binance Order IDs, and Strategy Targets are saved to **MongoDB**.
5.  **Monitoring**: The UI draws a **Visual Card** with a P&L progress bar. P&L is calculated using the real-time price against the recorded average entry price.

## 2. Quantum Scanner & AI Ranking 🤖

1.  **Discovery**: Backend fetches 24hr stats for all 300+ USDT/USDC pairs. It filters for symbols where `status == 'TRADING'` and `24h_volume > $1M`.
2.  **Scoring**: Top 20 candidates are analyzed across 7 timeframes. A **Quant Score (0-10)** is assigned based on RSI, ADX, and EMAs.
3.  **AI Filtering (Ranking)**: The user clicks "AI Rank." The current scanner table is sent to the local **Qwen 2.5 14B** model. The AI identifies the top 3 setups with the highest technical confluence.
4.  **Expert Deep Analysis**: Clicking the "Activity" icon on a row triggers a **Deep Dive**. The system rescans all timeframes for that specific coin and asks the AI to generate a precise trade setup (Entry, TP, SL).

## 3. Autonomous Reconciliation (The Reconciler) 🛡️

A background task running every 30 seconds to keep MongoDB in sync with Binance reality.

*   **Order ID Verification**: It queries Binance for every specific Order ID stored in a trade's `orders` array.
*   **Fill Detection**: If a TP or SL ID is missing from "Open Orders," it checks the **Binance Trade History**.
*   **Auto-Closure**: If a `FILLED` status is found, it captures the **Exit Price** and **Exit Fees**, marks the trade as `CLOSED` in MongoDB, and the card moves to the "Closed Trades" tab.
*   **Manual Control Detection**: If orders are `CANCELLED` (e.g., via the Binance app), it switches the trade to `MANUAL_CONTROL` mode in our UI.

## 4. System Auditing & Data Persistence 🗄️

*   **Request Auditing**: Every single POST/DELETE/GET interaction with the Binance API is "Black-Boxed" in MongoDB with its raw payload and raw response.
*   **Database**: All trades, rankings, and audit logs are stored in **MongoDB**. This ensures your trade history survives computer restarts and backend updates.
*   **Safety Guards**: The system proactively uses `recvWindow=60000` and automatic clock synchronization to prevent timestamp-related API rejections.
