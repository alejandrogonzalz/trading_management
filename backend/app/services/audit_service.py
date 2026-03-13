import time
from typing import Any, Dict
from app.db.database import audit_collection

def log_api_call(method: str, endpoint: str, request_data: Any, response_data: Any, status_code: int = 200):
    """Logs an API interaction to MongoDB audit collection."""
    entry = {
        "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
        "timestamp": time.time(),
        "method": method,
        "endpoint": endpoint,
        "request": request_data,
        "response": response_data,
        "status": status_code
    }
    
    try:
        audit_collection.insert_one(entry)
    except Exception as e:
        print(f"FAILED TO AUDIT TO MONGO: {e}")

def get_recent_logs(limit: int = 100):
    """Retrieves most recent logs from MongoDB."""
    cursor = audit_collection.find({}).sort("timestamp", -1).limit(limit)
    return list(cursor)
