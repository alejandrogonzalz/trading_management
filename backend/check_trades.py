import sqlite3
import json

conn = sqlite3.connect("data/trading.db")
cursor = conn.cursor()

# Get column names
cursor.execute("PRAGMA table_info(trades)")
columns = cursor.fetchall()
print("Columns in trades table:")
for col in columns:
    print(f"  {col[1]} ({col[2]})")

# Check closed trades with exit_price
cursor.execute("""
    SELECT id, symbol, status, entry_price, exit_price, quantity, tp, sl, orders
    FROM trades 
    WHERE status = 'CLOSED'
    ORDER BY timestamp DESC
    LIMIT 10
""")
closed = cursor.fetchall()
print(f"\nClosed trades ({len(closed)}):")
for row in closed:
    id_val, symbol, status, entry_price, exit_price, quantity, tp, sl, orders_json = row
    orders = json.loads(orders_json) if orders_json else []
    exit_orders = [o for o in orders if o.get("role") in ("TP", "SL", "MANUAL_EXIT")]
    print(
        f"  {id_val} - {symbol}: entry=${entry_price:.8f}, exit=${exit_price if exit_price else 'NULL'}, qty={quantity:.2f}, orders={len(orders)} (exit orders: {len(exit_orders)})"
    )

# Check ENTRY_FILLED trades that might be missing exit price
cursor.execute("""
    SELECT id, symbol, status, entry_price, exit_price, quantity, tp, sl, orders
    FROM trades 
    WHERE status = 'ENTRY_FILLED'
    ORDER BY timestamp DESC
    LIMIT 10
""")
entry_filled = cursor.fetchall()
print(f"\nENTRY_FILLED trades ({len(entry_filled)}):")
for row in entry_filled:
    id_val, symbol, status, entry_price, exit_price, quantity, tp, sl, orders_json = row
    orders = json.loads(orders_json) if orders_json else []
    print(
        f"  {id_val} - {symbol}: entry=${entry_price:.8f}, exit=${exit_price if exit_price else 'NULL'}, qty={quantity:.2f}, orders={len(orders)}"
    )

cursor.close()
conn.close()
