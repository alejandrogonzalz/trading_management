import os
from pymongo import MongoClient
from app.core.config import settings

# Use MONGO_URL from environment or default to local
MONGO_URL = os.getenv("MONGO_URL", "mongodb://localhost:27017")

# Synchronous client for current architecture
client = MongoClient(MONGO_URL)
db = client.trading_db

# Collections
trades_collection = db.trades
audit_collection = db.audit_log

def ping_db():
    try:
        client.admin.command('ping')
        print("✅ MongoDB Connection Successful")
        return True
    except Exception as e:
        print(f"❌ MongoDB Connection Failed: {e}")
        return False
