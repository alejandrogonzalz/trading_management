# Trading Management Workstation - Backend API Reference

This document outlines the available FastAPI endpoints used by the React frontend to manage trading, scanning, and auditing.

## 1. Trade Management (`/trades`)

### **GET /trades/open**
Fetches all currently active trades, including "Virtual Positions" (coins held in wallet) and pending OCO orders.
*   **Returns**: `List[Order]` (enriched with `smart_meta` from MongoDB).

### **POST /trades/smart-trade**
Executes a new "Smart Trade" (Market/Limit Entry + OCO Protectors).
*   **Request Body**: `SmartTradeRequest`
    *   `symbol` (str): e.g., "BTCUSDT"
    *   `quantity` (float)
    *   `take_profit_price` (float)
    *   `stop_loss_price` (float)
    *   `side` (str): "BUY" or "SELL"
    *   `mode` (str): "SPOT" or "LEAD"
*   **Response**: `{ "entry": Order, "exit_strategy": Order }`

### **GET /trades/smart-history**
Returns a list of all Smart Trades that have been closed, including calculated P&L and fees.
*   **Returns**: `List[TradeMetadata]` (sorted by most recent).

### **POST /trades/market-close**
Surgically closes a specific Smart Trade by cancelling its protection legs and executing an immediate Market Sell of its specific quantity.
*   **Request Body**: `MarketCloseRequest`
    *   `symbol` (str)
    *   `quantity` (Optional[float])
    *   `orderListId` (Optional[int])
*   **Response**: `Order` (The Market Sell confirmation).

### **DELETE /trades/order**
Cancels a single specific order on Binance.
*   **Request Body**: `CancelOrderRequest`
    *   `symbol` (str)
    *   `orderId` (int)

---

## 2. Market Data & Scanner

### **GET /scanner/table**
Returns the results of the last full market scan.
*   **Returns**: `{ "timestamp": str, "results": List[ScanRow] }`

### **POST /scanner/run**
Triggers a fresh scan of the top opportunity pairs.
*   **Parameters**: `RunScannerRequest` (Optional list of pairs).

### **GET /market/multi-timeframe-candles/{symbol}**
Fetches OHLCV data for a symbol across multiple timeframes (5m, 1h, 1d, etc.).

### **GET /indicators/{symbol}/{interval}**
Returns technical indicator values (RSI, ADX, MACD, etc.) for a specific symbol and timeframe.

---

## 3. Account & Auditing

### **GET /account/balances**
Returns the current wallet balances from Binance.
*   **Returns**: `List[Balance]` (Filtered for USDC/USDT and active holdings).

### **GET /audit/logs**
Retrieves the permanent record of every API interaction with Binance.
*   **Parameters**: `limit` (int, default 50).
*   **Returns**: `List[AuditEntry]` (Request payload + Raw Response).

### **GET /audit/ping-db**
Checks connectivity to the MongoDB database.
*   **Returns**: `{ "connected": bool }`
