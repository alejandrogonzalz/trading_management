import sqlite3
import sys

conn = sqlite3.connect("data/trading.db")
cursor = conn.cursor()

# Get column names
cursor.execute("PRAGMA table_info(trades)")
columns = [col[1] for col in cursor.fetchall()]
print("Columns:", columns)

# Query PEPE trades
cursor.execute(
    "SELECT * FROM trades WHERE symbol LIKE '%PEPE%' ORDER BY timestamp DESC"
)
rows = cursor.fetchall()

for row in rows:
    print("\n--- Trade ---")
    for i, col in enumerate(columns):
        print(f"{col}: {row[i]}")

    # Also check if there are any lead_trades
cursor.execute(
    "SELECT * FROM lead_trades WHERE symbol LIKE '%PEPE%' ORDER BY timestamp DESC"
)
lead_rows = cursor.fetchall()
if lead_rows:
    cursor.execute("PRAGMA table_info(lead_trades)")
    lead_columns = [col[1] for col in cursor.fetchall()]
    print("\n=== LEAD TRADES ===")
    for row in lead_rows:
        print("\n--- Lead Trade ---")
        for i, col in enumerate(lead_columns):
            print(f"{col}: {row[i]}")

cursor.close()
conn.close()
