"""Download historical OHLCV candles from Binance public API."""

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any

import httpx

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
DATA_DIR = Path(__file__).parent / "data" / "candles"

DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]


def _ms(dt_str: str) -> int:
    """Convert 'YYYY-MM-DD' to milliseconds since epoch."""
    return int(datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


async def fetch_candles(
    symbol: str,
    interval: str = "1h",
    start_date: str = "2025-11-01",
    end_date: str = "2026-05-01",
) -> List[Dict[str, Any]]:
    """Fetch historical candles from Binance, auto-paginating in 1000-candle chunks."""
    start_ms = _ms(start_date)
    end_ms = _ms(end_date)
    all_candles: List[Dict[str, Any]] = []

    async with httpx.AsyncClient(timeout=30) as client:
        current_ms = start_ms
        while current_ms < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_ms,
                "endTime": end_ms,
                "limit": 1000,
            }
            resp = await client.get(BINANCE_KLINES_URL, params=params)
            resp.raise_for_status()
            raw = resp.json()
            if not raw:
                break

            for k in raw:
                all_candles.append(
                    {
                        "timestamp": k[0],
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    }
                )

            # Advance past the last candle's open time
            current_ms = raw[-1][0] + 1
            # Small delay to respect rate limits
            await asyncio.sleep(0.1)

    return all_candles


async def fetch_multi_tf_candles(
    symbol: str,
    timeframes: List[str] = DEFAULT_TIMEFRAMES,
    start_date: str = "2025-11-01",
    end_date: str = "2026-05-01",
) -> Dict[str, List[Dict[str, Any]]]:
    """Fetch candles for multiple timeframes. Returns {tf: [candles]}."""
    result: Dict[str, List[Dict[str, Any]]] = {}
    for tf in timeframes:
        print(f"  Fetching {symbol} {tf}...")
        candles = await fetch_candles(symbol, tf, start_date, end_date)
        result[tf] = candles
        print(f"    {len(candles)} candles")
    return result


def save_candles(candles: List[Dict[str, Any]], symbol: str, interval: str) -> Path:
    """Save candles to JSON file."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{symbol}_{interval}.json"
    with open(path, "w") as f:
        json.dump(candles, f)
    return path
