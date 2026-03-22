from pymongo import MongoClient
import time
import sys
import os

# Add parent dir to path for imports if needed
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def check_orphaned_trades():
    client = MongoClient('mongodb://localhost:27017')
    db = client['trading_db']
    lead_trades = db.lead_trades
    
    orphaned = list(lead_trades.find({"status": "ENTRY_ONLY"}))
    
    if not orphaned:
        print("No orphaned ENTRY_ONLY trades found.")
        return

    print(f"Found {len(orphaned)} orphaned trades:")
    for t in orphaned:
        print(f"ID: {t['_id']} | Symbol: {t['symbol']} | Side: {t['side']} | Qty: {t['quantity']} | Time: {t['timestamp']}")
        
    confirm = input("\nDo you want to mark these as CLOSED in DB? (y/n): ")
    if confirm.lower() == 'y':
        res = lead_trades.update_many(
            {"status": "ENTRY_ONLY"},
            {"$set": {"status": "CLOSED", "close_reason": "MANUAL_CLEANUP", "close_time": time.time()}}
        )
        print(f"Updated {res.modified_count} trades.")
    else:
        print("Cleanup cancelled.")

if __name__ == "__main__":
    check_orphaned_trades()
