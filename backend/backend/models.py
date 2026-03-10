from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class SmartTradeRequest(BaseModel):
    symbol: str
    quantity: float
    buy_price: Optional[float] = None
    take_profit_price: float
    stop_loss_price: float

class CancelOrderRequest(BaseModel):
    symbol: str
    orderId: int

class MarketCloseRequest(BaseModel):
    symbol: str

class RunScannerRequest(BaseModel):
    pairs: List[str]
    timeframe: Optional[str] = "1h" # Default to 1h if not specified
