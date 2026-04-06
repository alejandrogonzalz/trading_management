import time
import json
from typing import Any, Dict
from app.db.database import db_session
from app.db.models import AuditLog
from pydantic import BaseModel

def log_api_call(method: str, endpoint: str, request_data: Any, response_data: Any, status_code: int = 200):
    """Logs an API interaction to SQLite audit collection."""
    
    # Robust serialization for SDK/Pydantic objects
    def serialize(obj):
        if isinstance(obj, BaseModel):
            return obj.model_dump()
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        if hasattr(obj, '__dict__'):
            try:
                return json.loads(json.dumps(obj, default=lambda o: o.__dict__))
            except:
                return str(obj)
        return str(obj)

    try:
        # Pre-process request and response to ensure they are serializable
        serializable_req = request_data
        if not isinstance(request_data, (dict, list, str, int, float, bool, type(None))):
            serializable_req = serialize(request_data)

        serializable_res = response_data
        if not isinstance(response_data, (dict, list, str, int, float, bool, type(None))):
            serializable_res = serialize(response_data)

        entry = AuditLog(
            timestamp_str=time.strftime("%Y-%m-%d %H:%M:%S"),
            timestamp=time.time(),
            method=method,
            endpoint=endpoint,
            request=serializable_req,
            response=serializable_res,
            status=status_code
        )
        
        db_session.add(entry)
        db_session.commit()
    except Exception as e:
        db_session.rollback()
        print(f"FAILED TO AUDIT TO SQLITE: {e}")
    finally:
        db_session.remove()

def get_recent_logs(limit: int = 100):
    """Retrieves most recent logs from SQLite."""
    try:
        logs = db_session.query(AuditLog).order_by(AuditLog.timestamp.desc()).limit(limit).all()
        # Convert to dict for API compatibility
        result = []
        for log in logs:
            result.append({
                "id": log.id,
                "timestamp_str": log.timestamp_str,
                "timestamp": log.timestamp,
                "method": log.method,
                "endpoint": log.endpoint,
                "request": log.request,
                "response": log.response,
                "status": log.status
            })
        return result
    finally:
        db_session.remove()
