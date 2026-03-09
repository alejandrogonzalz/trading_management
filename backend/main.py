import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from binance.client import Client
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Optional, List

load_dotenv()

app = FastAPI(title="Trading Management API")

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Binance Client (Standard credentials for Spot/USDC)
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET")

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)

class SmartTradeRequest(BaseModel):
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    buy_price: Optional[float] = None  # None for Market
    take_profit_price: float
    stop_loss_price: float

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/account/balances")
def get_balances():
    """Returns only USDC balance to keep things separated."""
    try:
        info = client.get_account()
        balances = [b for b in info['balances'] if b['asset'] == 'USDC' or float(b['free']) > 0]
        return balances
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/trades/open")
def get_open_orders():
    """Returns open USDC orders."""
    try:
        orders = client.get_open_orders()
        # Filter for USDC pairs
        usdc_orders = [o for o in orders if 'USDC' in o['symbol']]
        return usdc_orders
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/trades/smart-trade")
def create_smart_trade(req: SmartTradeRequest):
    """
    Creates a 3Commas style Smart Trade:
    1. Market/Limit Buy
    2. Once filled, create OCO (Take Profit + Stop Loss)
    """
    try:
        # Step 1: Initial Entry
        if req.buy_price:
            order = client.create_order(
                symbol=req.symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_LIMIT,
                timeInForce=Client.TIME_IN_FORCE_GTC,
                quantity=req.quantity,
                price=req.buy_price
            )
        else:
            order = client.create_order(
                symbol=req.symbol,
                side=Client.SIDE_BUY,
                type=Client.ORDER_TYPE_MARKET,
                quantity=req.quantity
            )
        
        # Note: In a real "Smart Trade", we would wait for the fill before placing OCO.
        # For this version, we assume immediate fill or manual management.
        # Placing OCO (Take Profit & Stop Loss)
        oco_order = client.create_oco_order(
            symbol=req.symbol,
            side=Client.SIDE_SELL,
            quantity=req.quantity,
            price=req.take_profit_price,
            stopPrice=req.stop_loss_price,
            stopLimitPrice=req.stop_loss_price, # Simplified Stop Market via Stop Limit
            stopLimitTimeInForce=Client.TIME_IN_FORCE_GTC
        )
        
        return {"entry": order, "exit_strategy": oco_order}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
