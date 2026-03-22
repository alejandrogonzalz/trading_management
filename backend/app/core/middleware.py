import time
import json
from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from app.db.database import db

class EndpointAuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        request_body = None
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Read body and then replace it so following handlers can read it again
                body = await request.body()
                if body:
                    request_body = json.loads(body)
                # Re-create request with the body so it's available for standard FastAPI parsing
                async def receive():
                    return {"type": "http.request", "body": body}
                request._receive = receive
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
            "process_time_ms": round(process_time * 1000, 2),
            "request_body": request_body
        }
        
        try:
            db.endpoint_audit.insert_one(entry)
        except Exception as e:
            print(f"ERROR logging endpoint audit: {e}")
            
        return response
