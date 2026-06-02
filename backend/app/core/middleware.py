import json
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from app.db.database import db_session
from app.db.models import EndpointAudit


class EndpointAuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        request_body = None
        # ... (rest of the body reading logic remains the same)
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Read body and then replace it so following handlers can read it again
                body = await request.body()
                if body:
                    try:
                        request_body = json.loads(body)
                    except:
                        request_body = {"raw": str(body)}

                # Re-create request with the body so it's available for standard FastAPI parsing
                async def receive():
                    return {"type": "http.request", "body": body}

                request._receive = receive
            except:
                pass

        response = await call_next(request)
        process_time = time.time() - start_time

        # Log the call
        try:
            audit_entry = EndpointAudit(
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                unix_time=time.time(),
                method=request.method,
                url=str(request.url),
                status_code=response.status_code,
                process_time_ms=round(process_time * 1000, 2),
                request_body=request_body,
            )
            db_session.add(audit_entry)
            db_session.commit()
        except Exception as e:
            db_session.rollback()
            print(f"ERROR logging endpoint audit: {e}")
        finally:
            db_session.remove()

        return response
