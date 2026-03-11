# Application Workflows

This document describes the high-level automated and manual processes that power the Trading Workstation.

## 1. The "Smart Trade" Lifecycle (Spot)

This is the primary manual trading flow designed for set-and-forget execution.

1.  **Entry Execution**: 
    *   User clicks "Execute" in the Smart Terminal.
    *   Backend executes a `MARKET` or `LIMIT` order on Binance.
    *   Once confirmed, the actual weighted average fill price is captured.
2.  **Protection Setup**:
    *   Backend immediately calculates and places an `OCO` (One-Cancels-the-Other) order.
    *   Upper Leg: `LIMIT_MAKER` (Take Profit).
    *   Lower Leg: `STOP_LOSS_LIMIT` (Stop Loss).
3.  **Local Persistence**:
    *   The trade ID, Entry Price, and OCO List ID are saved to MongoDB.
    *   The trade is marked as `ACTIVE`.
4.  **Monitoring**:
    *   The UI displays a **Smart Card** with a visual progress bar and real-time P&L.
5.  **Closing**:
    *   If targets are hit on Binance, the **Reconciler** (see below) detects it.
    *   If the user clicks **Panic Sell**, the backend cancels the OCO and executes a Market Sell.
    *   The trade is marked `CLOSED` and moved to **History**.

## 2. Autonomous Trade Reconciliation

The "Truth-Seeker" background task that ensures local data matches Binance reality.

*   **Frequency**: Runs every 30 seconds.
*   **Logic**:
    1.  Fetch all `ACTIVE` trades from MongoDB.
    2.  Check Binance for the specific `orderListId` or `orderId` associated with each trade.
    3.  If an ID is missing from "Open Orders", check Binance History.
    4.  **Branch A (FILL)**: If the leg is `FILLED`, the system captures the exit price/fees and archives the trade.
    5.  **Branch B (CANCEL)**: If the leg was cancelled manually on Binance, the system switches the trade to `MANUAL_CONTROL` mode.

## 3. The "Turbo-Scan" AI Pipeline

1.  **Market Filtering**: The system fetches 24hr statistics for all USDT/USDC pairs and selects the top 20 based on volatility and volume.
2.  **Multithreaded Scan**: Fetches 140+ technical data points across 7 timeframes in under 3 seconds.
3.  **AI Analysis (Qwen 2.5 14B)**:
    *   Passes the scan results to the local LLM.
    *   The LLM ranks the top 3 setups based on technical confluence.
    *   Generated Entry, TP, and SL targets are sent to the frontend for one-click execution.

## 4. API Audit Logging

Every interaction with the Binance API is "Black-Boxed" for safety.

*   **Capture**: The `audit_service` intercepts the request payload and the raw JSON response from Binance.
*   **Storage**: Logs are stored in the MongoDB `audit_log` collection.
*   **Purpose**: Allows the developer to troubleshoot failed trades by seeing exactly what the exchange rejected (e.g., Notional limits, precision errors).
