import sqlite3
import os
import sys

# Define path to the database
db_path = os.path.join(os.path.dirname(__file__), "backend", "data", "trading.db")
if not os.path.exists(db_path):
    # Try alternate path if running from within backend or script folder
    db_path = os.path.join(os.getcwd(), "backend", "data", "trading.db")

def delete_lead_history():
    if not os.path.exists(db_path):
        print(f"Error: Database not found at {db_path}")
        return

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Count closed lead trades before deletion
        cursor.execute("SELECT COUNT(*) FROM lead_trades WHERE status = 'CLOSED'")
        count = cursor.fetchone()[0]

        if count == 0:
            print("No closed Lead trades found in history.")
            conn.close()
            return

        print(f"Found {count} closed Lead trades. Deleting...")

        # Delete only CLOSED lead trades
        cursor.execute("DELETE FROM lead_trades WHERE status = 'CLOSED'")
        
        conn.commit()
        print(f"Successfully deleted {count} Smart Lead history records.")
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM lead_trades WHERE status = 'CLOSED'")
        remaining = cursor.fetchone()[0]
        print(f"Remaining closed Lead trades: {remaining}")

        conn.close()
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    delete_lead_history()
