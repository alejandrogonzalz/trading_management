# Trading Management Workstation - Full API Reference

This document provides a comprehensive technical reference for all FastAPI endpoints.

> **Status Key:**
> ✅ **TESTED**: Verified via Integration Tests.
> 🛡️ **STABLE**: Production-ready with error handling.
> 🤖 **AI-DRIVEN**: Interfaces with local LLM.

## 1. Trade Management (`/trades`)

### **GET /trades/open** ✅ 🛡️
Returns all active setups.
*   **Logic**: Aggregates Binance open orders + local MongoDB metadata + "Virtual Positions" (wallet holdings).
*   **Response**: `List[Order]` enriched with `smart_meta`.

### **POST /trades/smart-trade** ✅ 🛡️
The primary entry execution engine.
*   **Body**: `SmartTradeRequest(symbol, quantity, tp, sl, side, mode)`
*   **Workflow**: Market Buy -> Capture Fill Price -> Calculate Fee-Adjusted Qty -> Place OCO Protectors -> Save to MongoDB.

### **GET /trades/smart-history** ✅ 🛡️
The P&L engine.
*   **Returns**: All closed trades from MongoDB with entry/exit prices, timestamps, and fees.

### **POST /trades/market-close** ✅ 🛡️
Surgical "Panic Sell."
*   **Body**: `MarketCloseRequest(symbol, quantity, orderListId)`
*   **Workflow**: Cancel specific OCO -> Wait 1.5s -> Market Sell specific Qty -> Mark CLOSED in DB.

### **DELETE /trades/order** ✅
Cancels any single order on Binance.

---

## 2. Quantum Scanner (`/scanner`)

### **GET /scanner/table** ✅
Fetches the results of the last automated or manual scan.

### **POST /scanner/run** ✅ 🛡️
Triggers the multithreaded scanning engine.
*   **Body**: `RunScannerRequest(pairs, timeframe)`
*   **Logic**: Proactively filters for `status == 'TRADING'`. Scans across multiple timeframes.

---

## 3. AI & Analysis (`/llm`)

### **POST /llm/rank** 🤖
Triggers the Qwen 2.5 14B model to analyze the current scanner table.
*   **Returns**: Sorted list of top 3 high-probability setups with AI Bias and Reasoning.

### **GET /llm/analyze_row/{symbol}** 🤖
Performs a "Deep Analysis" on a single coin.
*   **Logic**: Fresh 7-timeframe rescan -> LLM context generation -> Target price (Entry/TP/SL) generation.

---

## 4. Market Data & Indicators (`/market`, `/indicators`, `/score`)

### **GET /market/multi-timeframe-candles/{symbol}** ✅
Fetches OHLCV data for 5m, 15m, 1h, 4h, 1d, 1w, 1M timeframes in parallel.

### **GET /indicators/{symbol}/{interval}** ✅
Calculates RSI, ADX, MACD, Bollinger Bands, and EMA Ribbon values.

### **GET /score/{symbol}/{interval}** ✅
Returns a 0-10 Quant Confluence score based on technical alignment.

---

## 5. Account & System (`/account`, `/audit`)

### **GET /account/balances** ✅ 🛡️
Returns live wallet balances. Automatically filters out exchange dust.

### **GET /audit/logs** ✅
Returns the "Black Box" log of all Binance API interactions from MongoDB.

### **GET /audit/ping-db** ✅
Verifies MongoDB connectivity.
