import sqlite3
import os
import sys

db_path = os.path.join(os.path.dirname(__file__), "data", "trading.db")

if not os.path.exists(db_path):
    print(f"Database not found at {db_path}")
    sys.exit(1)

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row
cursor = conn.cursor()

# Check if lead_trades table exists
cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name='lead_trades'"
)
if cursor.fetchone():
    print("Table 'lead_trades' exists.")
    cursor.execute("SELECT * FROM lead_trades")
    rows = cursor.fetchall()
    print(f"Found {len(rows)} rows in lead_trades:")
    for row in rows:
        print(dict(row))
else:
    print("Table 'lead_trades' does not exist.")

# Also check the columns in the table
if len(rows) > 0:
    cursor.execute("PRAGMA table_info(lead_trades)")
    columns = cursor.fetchall()
    print("\nColumns in lead_trades:")
    for col in columns:
        print(f"  {col[1]} ({col[2]})")

conn.close()
