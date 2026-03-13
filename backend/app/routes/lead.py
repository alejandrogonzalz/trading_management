from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any

router = APIRouter(tags=["Lead"])

@router.get("/status")
def get_lead_status():
    return {"status": "NOT_IMPLEMENTED", "message": "Lead Trading integration coming soon."}
