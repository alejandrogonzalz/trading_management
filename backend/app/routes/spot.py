from fastapi import APIRouter, HTTPException
from typing import Optional, List, Dict, Any

# Local imports
from app.services import binance_service
from app.services import audit_service
from app.models import SmartTradeRequest, CancelOrderRequest, MarketCloseRequest

router = APIRouter(tags=["Spot"])

@router.get("/audit/logs")
def get_audit_logs(limit: int = 50):
    # MongoDB documents need _id converted to string for JSON serialization
    logs = audit_service.get_recent_logs(limit)
    for log in logs:
        log["_id"] = str(log["_id"])
    return logs

@router.get("/account/balances")
def get_balances():
    return binance_service.get_balances()

@router.get("/trades/open")
def get_open_orders():
    return binance_service.get_open_orders()

@router.post("/trades/smart-trade")
def create_smart_trade(req: SmartTradeRequest):
    return binance_service.create_smart_trade(
        symbol=req.symbol,
        quantity=req.quantity,
        buy_price=req.buy_price,
        take_profit_price=req.take_profit_price,
        stop_loss_price=req.stop_loss_price,
        side=req.side,
        mode=req.mode
    )

@router.get("/trades/history")
def get_trade_history(symbol: Optional[str] = None):
    return binance_service.get_trade_history(symbol=symbol)

@router.get("/trades/smart-history")
def get_smart_history():
    return binance_service.get_smart_history()

@router.delete("/trades/order")
def cancel_single_order(req: CancelOrderRequest):
    return binance_service.cancel_order(symbol=req.symbol, order_id=req.orderId)

@router.post("/trades/market-close")
def market_close(req: MarketCloseRequest):
    return binance_service.market_close_position(
        symbol=req.symbol,
        quantity=req.quantity,
        order_list_id=req.orderListId
    )
