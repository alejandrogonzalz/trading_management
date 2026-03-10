from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
import logging
import asyncio

# Local imports
from .config import settings
from .models import SmartTradeRequest, CancelOrderRequest, MarketCloseRequest, RunScannerRequest
from . import binance_service
from . import market_service
from . import market_service_utils
from . import indicator_service
from . import scoring_service
from . import scanner_service
from . import llm_service

# Configure logging for APScheduler
logging.basicConfig(level=logging.INFO)
logging.getLogger('apscheduler').setLevel(logging.INFO)

app = FastAPI(title="Trading Management API")
scheduler = AsyncIOScheduler()

# Setup CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Scheduler Jobs ---
async def scheduled_scan_job():
    """Background job to run the scanner for a predefined set of pairs."""
    print("Running scheduled scan job...")
    try:
        # Dynamically find the top 20 opportunity pairs
        top_pairs = await asyncio.to_thread(market_service_utils.get_top_opportunity_pairs, 20)
        await asyncio.to_thread(scanner_service.run_scan, top_pairs)
        print("Scheduled scan completed.")
    except Exception as e:
        print(f"Error during scheduled scan: {e}")

# --- FastAPI Lifespan Events ---
@app.on_event("startup")
async def startup_event():
    scheduler.start()
    scheduler.add_job(
        scheduled_scan_job, 
        IntervalTrigger(minutes=settings.SCANNER_INTERVAL_MINUTES),
        id='scheduled_scanner',
        replace_existing=True
    )
    print(f"Scheduler started. Scan job runs every {settings.SCANNER_INTERVAL_MINUTES} minutes.")

@app.on_event("shutdown")
async def shutdown_event():
    scheduler.shutdown()
    print("Scheduler shut down.")

# --- API Endpoints ---
@app.get("/symbols")
def get_symbols():
    return binance_service.get_symbols()

@app.get("/account/balances")
def get_balances():
    return binance_service.get_balances()

@app.get("/trades/open")
def get_open_orders():
    return binance_service.get_open_orders()

@app.post("/trades/smart-trade")
def create_smart_trade(req: SmartTradeRequest):
    return binance_service.create_smart_trade(
        symbol=req.symbol,
        quantity=req.quantity,
        buy_price=req.buy_price,
        take_profit_price=req.take_profit_price,
        stop_loss_price=req.stop_loss_price
    )

@app.get("/trades/history")
def get_trade_history(symbol: Optional[str] = None):
    return binance_service.get_trade_history(symbol=symbol)

@app.delete("/trades/order")
def cancel_single_order(req: CancelOrderRequest):
    return binance_service.cancel_order(symbol=req.symbol, order_id=req.orderId)

@app.post("/trades/market-close")
def market_close_position(req: MarketCloseRequest):
    return binance_service.market_close_position(symbol=req.symbol)

@app.get("/market/candles/{symbol}/{interval}")
def get_market_candles(symbol: str, interval: str):
    return market_service.get_candles(symbol, interval)

@app.get("/market/multi-timeframe-candles/{symbol}")
def get_market_multi_timeframe(symbol: str):
    return market_service.get_multi_timeframe_candles(symbol)

@app.get("/indicators/{symbol}/{interval}")
def get_indicators(symbol: str, interval: str):
    candle_data = market_service.get_candles(symbol, interval)
    return indicator_service.calculate_indicators(candle_data)

@app.get("/score/{symbol}/{interval}")
def get_score(symbol: str, interval: str):
    candle_data = market_service.get_candles(symbol, interval)
    indicators = indicator_service.calculate_indicators(candle_data)
    return scoring_service.calculate_score(indicators)

@app.post("/scanner/run")
async def run_scanner_api(req: RunScannerRequest):
    pairs = req.pairs
    base_tf = req.timeframe or "1h"
    if not pairs:
        pairs = await asyncio.to_thread(market_service_utils.get_top_opportunity_pairs, 20)
    results = await asyncio.to_thread(scanner_service.run_scan, pairs, base_tf)
    return results

@app.get("/scanner/table")
def get_scanner_table():
    return scanner_service.get_latest_scan()

@app.post("/llm/rank")
async def rank_llm_setups():
    latest_scan = scanner_service.get_latest_scan()
    scanner_table = latest_scan.get("results", [])
    if not scanner_table:
        raise HTTPException(status_code=400, detail="No scan results available to rank. Run scanner first.")
    return await llm_service.rank_setups(scanner_table)

@app.get("/llm/analyze_row/{symbol}")
async def analyze_llm_row(symbol: str):
    # Perform a fresh, targeted scan for this specific pair across all TFs
    print(f"Performing fresh multi-TF scan for {symbol} deep analysis...")
    multi_timeframe_candles = await asyncio.to_thread(market_service.get_multi_timeframe_candles, symbol)
    
    multi_timeframe_indicators = {}
    for interval, candles in multi_timeframe_candles.items():
        if candles:
            multi_timeframe_indicators[interval] = indicator_service.calculate_indicators(candles)
    
    return await llm_service.analyze_row({"symbol": symbol, "indicators": multi_timeframe_indicators})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
