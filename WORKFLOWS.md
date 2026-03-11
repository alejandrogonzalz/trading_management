# Application Workflows

This document describes the high-level automated and manual processes that power the Trading Workstation.

## 1. The "Smart Trade" Lifecycle (Spot) ✅ VERIFIED

This is the primary manual trading flow designed for set-and-forget execution.

1.  **Entry Execution**: 
    *   User clicks "Execute" in the Smart Terminal.
    *   Backend executes a `MARKET` or `LIMIT` order on Binance.
    *   **Validated**: Correct average fill price is captured even for split market fills.
2.  **Protection Setup**:
    *   Backend calculates OCO quantity by subtracting commissions (**Fee-Aware Clipping**).
    *   Places an `OCO` (One-Cancels-the-Other) order on Binance.
    *   **Validated**: Handles "Above/Below Type" mandatory parameters for latest Binance API.
3.  **Local Persistence**:
    *   The trade ID, Entry Price, and OCO List ID are saved to **MongoDB**.
4.  **Monitoring**:
    *   The UI displays a **Smart Card** with a visual progress bar.
    *   **Validated**: Multiple trades for the same coin are isolated and shown as separate cards.
5.  **Closing**:
    *   **Panic Sell**: Cancels the OCO and executes a Market Sell of the *exact* trade quantity.
    *   **Verified**: Does not affect other trades of the same asset.

## 2. Autonomous Trade Reconciliation (The "Truth-Seeker") ✅ VERIFIED

Runs every 30 seconds to ensure MongoDB matches Binance exchange reality.

*   **Logic**:
    1.  Fetch all `ACTIVE` trades from MongoDB.
    2.  Check Binance for the specific **Order IDs** (not just balance guesses).
    3.  If IDs are missing, check Order History.
    4.  **FILL Detection**: If ID is `FILLED`, the system captures the `exit_price` and `fees` and moves the trade to **History**.
    5.  **MANUAL Detection**: If ID is `CANCELLED`, the trade stays active but moves to `MANUAL_CONTROL` mode.

## 3. The "Turbo-Scan" AI Pipeline ✅ VERIFIED

1.  **Market Filtering**: Fetches 24hr statistics and selects top 20 opportunity pairs.
2.  **TRADING Guard**: Automatically skips any coin in `BREAK` or `HALT` status to prevent API errors.
3.  **Multithreaded Scan**: Fetches 140+ technical data points across 7 timeframes.
4.  **AI Analysis**: Local LLM (Qwen 2.5 14B) ranks setups and generates Suggested Entry/TP/SL.

## 4. API Audit Logging ✅ VERIFIED

*   **Logic**: Every interaction with the Binance API is "Black-Boxed" in MongoDB.
*   **Purpose**: Debugging notional errors, precision filters, and exchange-side market closures.
