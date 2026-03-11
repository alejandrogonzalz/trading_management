# Trading Management Workstation - Backend API Reference

This document outlines the available FastAPI endpoints used by the React frontend to manage trading, scanning, and auditing.

> **Status Key:**
> ✅ **TESTED**: Verified via Integration Tests (`test_integration_smart_trade.py`, `test_reconciler_fill.py`).
> 🛡️ **STABLE**: Robustly handles errors, dust, and Binance filter requirements.

## 1. Trade Management (`/trades`)

### **GET /trades/open** ✅ 🛡️
Fetches all currently active trades, including "Virtual Positions" (coins held in wallet) and pending OCO orders.
*   **Returns**: `List[Order]` (enriched with `smart_meta` from MongoDB).
*   **Tested Path**: Successfully groups multiple OCO legs into a single master trade ID.

### **POST /trades/smart-trade** ✅ 🛡️
Executes a new "Smart Trade" (Market/Limit Entry + OCO Protectors).
*   **Request Body**: `SmartTradeRequest`
    *   `symbol` (str): e.g., "BTCUSDT"
    *   `quantity` (float)
    *   `take_profit_price` (float)
    *   `stop_loss_price` (float)
    *   `side` (str): "BUY" (Long) or "SELL" (Short)
    *   `mode` (str): "SPOT" or "LEAD"
*   **Response**: `{ "entry": Order, "status": str }`
*   **Tested Path**: Market Buy -> Capture average fill price -> Calculate precise OCO qty (fee adjusted) -> Place OCO protectors.

### **GET /trades/smart-history** ✅ 🛡️
Returns a list of all Smart Trades that have been closed, including calculated P&L and fees.
*   **Returns**: `List[TradeMetadata]` (sorted by most recent).
*   **Tested Path**: Verified extraction of exact `exit_price` and `exit_fees` from Binance filled orders.

### **POST /trades/market-close** ✅ 🛡️
Surgically closes a specific Smart Trade by cancelling its protection legs and executing an immediate Market Sell of its specific quantity.
*   **Request Body**: `MarketCloseRequest`
    *   `symbol` (str)
    *   `quantity` (Optional[float])
    *   `orderListId` (Optional[int])
    *   `clientOrderId` (Optional[str])
*   **Response**: `Order` (The Market Sell confirmation).
*   **Tested Path**: Isolated closure verified (Closing trade A does not affect trade B).

### **DELETE /trades/order** ✅
Cancels a single specific order on Binance.
*   **Request Body**: `CancelOrderRequest`
    *   `symbol` (str)
    *   `orderId` (int)

---

## 2. Market Data & Scanner

### **GET /scanner/table** ✅
Returns the results of the last full market scan.
*   **Returns**: `{ "timestamp": str, "results": List[ScanRow] }`

### **POST /scanner/run** ✅
Triggers a fresh scan of the top opportunity pairs. 
*   **Logic**: Proactively filters out symbols not in `TRADING` status (Hides coins in maintenance/BREAK).
*   **Parameters**: `RunScannerRequest` (Optional list of pairs).

### **GET /symbols** ✅
Returns a list of all USDT/USDC pairs currently in `TRADING` status.

---

## 3. Account & Auditing

### **GET /account/balances** ✅
Returns the current wallet balances from Binance.
*   **Protection**: Automatically filters out "Dust" (very small balances).

### **GET /audit/logs** ✅
Retrieves the permanent record of every API interaction with Binance from MongoDB.
