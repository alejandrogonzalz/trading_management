from pydantic import BaseModel
from typing import Optional, List, Dict, Any

class SmartTradeRequest(BaseModel):
    symbol: str
    quantity: float
    buy_price: Optional[float] = None
    take_profit_price: float
    stop_loss_price: float
    side: Optional[str] = "BUY"
    mode: Optional[str] = "SPOT"

class CancelOrderRequest(BaseModel):
    symbol: str
    orderId: int

class MarketCloseRequest(BaseModel):
    symbol: str
    quantity: Optional[float] = None
    orderListId: Optional[int] = None

class RunScannerRequest(BaseModel):
    pairs: List[str]
    timeframe: Optional[str] = "1h" # Default to 1h if not specified

# --- Lead / Futures Models ---

class LeadOrderRequest(BaseModel):
    symbol: str
    side: str # 'BUY' or 'SELL'
    type: str # 'LIMIT' or 'MARKET'
    quantity: float
    price: Optional[float] = None
    leverage: Optional[int] = 10
    take_profit_price: Optional[float] = 0
    stop_loss_price: Optional[float] = 0

class SetLeverageRequest(BaseModel):
    symbol: str
    leverage: int

class FuturesCloseRequest(BaseModel):
    symbol: str
    quantity: Optional[float] = None
