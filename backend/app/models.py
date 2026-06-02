from pydantic import BaseModel


class SmartTradeRequest(BaseModel):
    symbol: str
    quantity: float
    buy_price: float | None = None
    take_profit_price: float
    stop_loss_price: float
    side: str | None = "BUY"
    mode: str | None = "SPOT"


class CancelOrderRequest(BaseModel):
    symbol: str
    orderId: int


class MarketCloseRequest(BaseModel):
    symbol: str
    quantity: float | None = None
    orderListId: int | None = None


class RunScannerRequest(BaseModel):
    pairs: list[str]
    timeframe: str | None = "1h"  # Default to 1h if not specified


# --- Lead / Futures Models ---


class LeadOrderRequest(BaseModel):
    symbol: str
    side: str  # 'BUY' or 'SELL'
    type: str  # 'LIMIT' or 'MARKET'
    quantity: float
    price: float | None = None
    leverage: int | None = 10
    take_profit_price: float | None = 0
    stop_loss_price: float | None = 0


class SetLeverageRequest(BaseModel):
    symbol: str
    leverage: int


class FuturesCloseRequest(BaseModel):
    symbol: str
    quantity: float | None = None
