import asyncio

from fastapi import APIRouter, HTTPException

from app.db import database
from app.models import RunScannerRequest

# Local imports
from app.services import (
    binance_service,
    indicator_service,
    llm_service,
    market_service,
    scanner_service,
    scoring_service,
)
from app.utils import market_utils as market_service_utils

router = APIRouter(tags=["Market"])


@router.get("/audit/ping-db")
def ping_db():
    return {"connected": database.ping_db()}


@router.get("/symbols")
def get_symbols():
    return binance_service.get_symbols()


@router.get("/market/candles/{symbol}/{interval}")
def get_market_candles(symbol: str, interval: str):
    return market_service.get_candles(symbol, interval)


@router.get("/market/multi-timeframe-candles/{symbol}")
def get_market_multi_timeframe(symbol: str):
    return market_service.get_multi_timeframe_candles(symbol)


@router.get("/indicators/{symbol}/{interval}")
def get_indicators(symbol: str, interval: str):
    candle_data = market_service.get_candles(symbol, interval)
    return indicator_service.calculate_indicators(candle_data)


@router.get("/score/{symbol}/{interval}")
def get_score(symbol: str, interval: str):
    candle_data = market_service.get_candles(symbol, interval)
    indicators = indicator_service.calculate_indicators(candle_data)
    return scoring_service.calculate_score(indicators)


@router.post("/scanner/run")
async def run_scanner_api(req: RunScannerRequest):
    # JIT WARMUP: Start loading the LLM immediately
    try:
        asyncio.create_task(llm_service.warm_up_llm())
        print("✅ Warmup task scheduled from API route.")
    except Exception as e:
        print(f"Warmup Trigger Error: {e}")

    pairs = req.pairs
    base_tf = req.timeframe or "1h"
    if not pairs:
        pairs = await asyncio.to_thread(market_service_utils.get_top_opportunity_pairs, 20)
    await asyncio.to_thread(scanner_service.run_scan, pairs, base_tf)
    return scanner_service.get_latest_scan()


@router.get("/scanner/table")
def get_scanner_table():
    return scanner_service.get_latest_scan()


@router.post("/llm/rank")
async def rank_llm_setups():
    latest_scan = scanner_service.get_latest_scan()
    scanner_table = latest_scan.get("results", [])
    if not scanner_table:
        raise HTTPException(status_code=400, detail="No scan results available to rank. Run scanner first.")
    return await llm_service.rank_setups(scanner_table)


@router.get("/llm/analyze_row/{symbol}")
async def analyze_llm_row(symbol: str, mode: str = "SPOT", use_langgraph: bool = True):
    # Perform a fresh, targeted scan for this specific pair across all TFs
    print(f"Performing fresh multi-TF scan for {symbol} deep analysis ({mode})...")
    multi_timeframe_candles = await asyncio.to_thread(market_service.get_multi_timeframe_candles, symbol)

    multi_timeframe_indicators = {}
    for interval, candles in multi_timeframe_candles.items():
        if candles:
            multi_timeframe_indicators[interval] = indicator_service.calculate_indicators(candles)

    if use_langgraph:
        return await llm_service.get_deep_langgraph_analysis(symbol, multi_timeframe_indicators, mode)

    # Fallback to legacy single-node analysis
    return await llm_service.analyze_row({"symbol": symbol, "indicators": multi_timeframe_indicators, "mode": mode})
