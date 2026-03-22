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
    print("\n--- SEARCHING FOR 'lead' IN AUDIT LOG ---")
    al = list(db.audit_log.find({"endpoint": {"$regex": "lead"}}).sort("unix_time", -1).limit(10))
    print(json.dumps(al, indent=2, default=json_util.default))

    print("\n--- ALL COLLECTIONS ---")
    print(db.list_collection_names())

if __name__ == "__main__":
    check_data()
