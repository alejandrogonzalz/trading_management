import time
import json
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.db.database import db

class EndpointAuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Capture request body if possible
        request_body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # We need to be careful with reading the body as it can only be read once
                # But for our small-scale app it should be fine if we use a specific approach
                # Actually, reading body here might interfere with FastAPI's own reading.
                # A safer way is to just log headers and query params for now, 
                # or use a more advanced body-replaying technique.
                pass
            except:
                pass

        response = await call_next(request)
        process_time = time.time() - start_time
        
        # Log the call
        entry = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "unix_time": time.time(),
            "method": request.method,
            "url": str(request.url),
            "client_host": request.client.host if request.client else None,
            "status_code": response.status_code,
            "process_time_ms": round(process_time * 1000, 2)
        }
        
        try:
            db.endpoint_audit.insert_one(entry)
        except Exception as e:
            print(f"ERROR logging endpoint audit: {e}")
            
        return response
