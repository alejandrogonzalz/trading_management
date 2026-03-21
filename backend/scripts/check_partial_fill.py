import os
import sys
from pymongo import MongoClient
import json
from bson import json_util

# Set PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")
client = MongoClient(MONGO_URL)
db = client.trading_db

def check_data():
    print("\n--- LATEST 3 LEAD TRADES ---")
    lead_trades = list(db.lead_trades.find().sort("timestamp", -1).limit(3))
    print(json.dumps(lead_trades, indent=2, default=json_util.default))

    print("\n--- LATEST 5 FAILED AUDIT LOGS ---")
    failed_logs = list(db.audit_log.find({"status": 400}).sort("unix_time", -1).limit(5))
    print(json.dumps(failed_logs, indent=2, default=json_util.default))

if __name__ == "__main__":
    check_data()
