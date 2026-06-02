import sqlite3

import requests

print("=== Final System Check ===")
# 1. Open orders endpoint
resp = requests.get("http://localhost:8001/trades/open")
if resp.status_code == 200:
    data = resp.json()
    print(f"Open orders count: {len(data)}")
    position_entries = [o for o in data if o.get("type") == "POSITION"]
    print(f"POSITION entries: {len(position_entries)}")
    for pos in position_entries:
        print(f"  {pos['symbol']}: origQty={pos.get('origQty')}, price={pos.get('price')}")
else:
    print(f"Error fetching open orders: {resp.status_code}")

# 2. Database check for corrupted trades
conn = sqlite3.connect("data/trading.db")
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'PROTECTION_FAILED'")
failed = cursor.fetchone()[0]
print(f"\nPROTECTION_FAILED trades: {failed}")
cursor.execute("SELECT COUNT(*) FROM trades WHERE status = 'MANUAL_CONTROL'")
manual = cursor.fetchone()[0]
print(f"MANUAL_CONTROL trades: {manual}")
cursor.close()
conn.close()

# 3. Formatting test
from app.services.binance_service import format_price

pepe_price = 0.00000358
formatted = format_price("PEPEUSDT", pepe_price)
print(f"\nPEPE price formatting: {pepe_price} -> {formatted}")
print("All checks completed.")
