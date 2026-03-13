from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any

from app.services.futures_service import futures_service
from app.models import LeadOrderRequest, SetLeverageRequest, FuturesCloseRequest, CancelOrderRequest

router = APIRouter(tags=["Lead Trading"])

@router.get("/status")
def get_lead_status():
    """Returns Lead Trader status and account info."""
    return futures_service.get_lead_status()

@router.get("/symbols")
def get_lead_symbols():
    """Returns the list of symbols whitelisted for Lead Trading."""
    return futures_service.get_tradable_symbols()

@router.get("/balances")
def get_lead_balances():
    """Retrieves USDS-M Futures wallet balances."""
    return futures_service.get_balances()

@router.get("/positions")
def get_lead_positions(symbol: Optional[str] = None):
    """Retrieves active USDS-M Futures lead positions."""
    return futures_service.get_active_positions(symbol=symbol)

@router.get("/open-orders")
def get_lead_open_orders(symbol: Optional[str] = None):
    """Retrieves pending lead orders."""
    return futures_service.get_open_orders(symbol=symbol)

@router.get("/history")
def get_lead_history():
    """Retrieves closed Lead/Futures trades from MongoDB."""
    return futures_service.get_lead_history()

@router.get("/binance-history")
def get_lead_binance_history(symbol: Optional[str] = None):
    """Retrieves raw execution history from USDS-M Futures."""
    return futures_service.get_binance_trade_history(symbol=symbol)

@router.delete("/order")
def cancel_lead_order(req: CancelOrderRequest):
    """Cancels a pending Lead order."""
    return futures_service.cancel_order(symbol=req.symbol, order_id=req.orderId)

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

@router.post("/smart-order")
def create_smart_lead_order(req: LeadOrderRequest):
    """Places a new lead order with automated TP/SL management."""
    return futures_service.create_smart_lead_order(
        symbol=req.symbol,
        side=req.side,
        quantity=req.quantity,
        tp_price=req.take_profit_price or 0,
        sl_price=req.stop_loss_price or 0,
        leverage=req.leverage or 10
    )

@router.post("/close-position")
def close_lead_position(req: FuturesCloseRequest):
    """Closes an active lead position via market order."""
    return futures_service.close_position(req.symbol, req.quantity)
