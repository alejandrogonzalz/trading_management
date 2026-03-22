import time
import json
from typing import Any, Dict
from app.db.database import audit_collection
from pydantic import BaseModel

def log_api_call(method: str, endpoint: str, request_data: Any, response_data: Any, status_code: int = 200):
    """Logs an API interaction to MongoDB audit collection."""
    
    # Robust serialization for SDK/Pydantic objects
    def serialize(obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dict__'):
            return obj.__dict__
        return str(obj)

    try:
        # Pre-process request and response to ensure they are BSON-serializable
        serializable_req = request_data
        if not isinstance(request_data, (dict, list, str, int, float, bool, type(None))):
            serializable_req = serialize(request_data)
        elif isinstance(request_data, dict):
            serializable_req = {k: (serialize(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) else v) for k, v in request_data.items()}

        serializable_res = response_data
        if not isinstance(response_data, (dict, list, str, int, float, bool, type(None))):
            serializable_res = serialize(response_data)
        elif isinstance(response_data, dict):
            serializable_res = {k: (serialize(v) if not isinstance(v, (dict, list, str, int, float, bool, type(None))) else v) for k, v in response_data.items()}

        entry = {
            "timestamp_str": time.strftime("%Y-%m-%d %H:%M:%S"),
            "timestamp": time.time(),
            "method": method,
            "endpoint": endpoint,
            "request": serializable_req,
            "response": serializable_res,
            "status": status_code
        }
        
        audit_collection.insert_one(entry)
    except Exception as e:
        print(f"FAILED TO AUDIT TO MONGO: {e}")

def get_recent_logs(limit: int = 100):
    """Retrieves most recent logs from MongoDB."""
    cursor = audit_collection.find({}).sort("timestamp", -1).limit(limit)
    return list(cursor)
