
---

## 6. Database Schema & Audit

The system uses a persistent MongoDB auditing architecture to track every trade lifecycle and API interaction.

### **Collection: `lead_trades`**
Stores the state of active "Virtual" Lead/Futures positions.
```json
{
  "_id": "LEAD_1773287604",
  "symbol": "BTCUSDT",
  "side": "SELL",
  "entry_price": 65000.50,
  "quantity": 0.005,
  "tp": 64000.00,
  "sl": 66000.00,
  "leverage": 10,
  "status": "ACTIVE",  // "ACTIVE" or "CLOSED"
  "timestamp": 1773287604.202,
  "entry_order_id": 945158985
}
```

### **Collection: `audit_log`**
The primary "Black Box" log for all external API calls to Binance (Spot/Futures).
```json
{
  "timestamp_str": "2026-03-12 03:53:24",
  "timestamp": 1773287604.202,
  "method": "POST",
  "endpoint": "lead/smart-entry",
  "request": { "symbol": "BTCUSDT", "side": "SELL", ... },
  "response": { "orderId": 945158985, "status": "FILLED", ... },
  "status": 200
}
```

### **Collection: `endpoint_audit`**
Logs all internal HTTP requests between the Frontend and Backend.
```json
{
  "timestamp": "2026-03-12 03:53:24",
  "unix_time": 1773287604.202,
  "method": "POST",
  "url": "http://localhost:8001/lead/smart-order",
  "status_code": 200,
  "process_time_ms": 150.5,
  "request_body": { ... } // Captured if POST/PUT
}
```