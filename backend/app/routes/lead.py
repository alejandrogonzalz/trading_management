from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any

from app.services.futures_service import futures_service
from app.models import LeadOrderRequest, SetLeverageRequest, FuturesCloseRequest

router = APIRouter(tags=["Lead Trading"])

@router.get("/status")
def get_lead_status():
    """Returns Lead Trader status and account info."""
    return futures_service.get_lead_status()

@router.get("/symbols")
def get_lead_symbols():
    """Returns the list of symbols whitelisted for Lead Trading."""
    return futures_service.get_tradable_symbols()

@router.get("/positions")
def get_lead_positions(symbol: Optional[str] = None):
    """Retrieves active USDS-M Futures lead positions."""
    return futures_service.get_active_positions(symbol=symbol)

@router.get("/open-orders")
def get_lead_open_orders(symbol: Optional[str] = None):
    """Retrieves pending lead orders."""
    return futures_service.get_open_orders(symbol=symbol)

@router.post("/leverage")
def set_lead_leverage(req: SetLeverageRequest):
    """Sets initial leverage for a lead trading symbol."""
    return futures_service.set_leverage(req.symbol, req.leverage)

@router.post("/order")
def create_lead_order(req: LeadOrderRequest):
    """Places a new lead order that will be copied by followers."""
    return futures_service.create_lead_order(
        symbol=req.symbol,
        side=req.side,
        order_type=req.type,
        quantity=req.quantity,
        price=req.price
    )

@router.post("/close-position")
def close_lead_position(req: FuturesCloseRequest):
    """Closes an active lead position via market order."""
    return futures_service.close_position(req.symbol, req.quantity)
