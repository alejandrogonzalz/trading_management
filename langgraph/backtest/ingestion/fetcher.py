"""Download historical OHLCV candles from Binance public API."""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from backtest.config import DEFAULT_TIMEFRAMES

BINANCE_KLINES_URL = "https://api.binance.com/api/v3/klines"
DATA_DIR = Path(__file__).parent.parent / "data" / "candles"

_PAGE_LIMIT = 1000
_TF_CONCURRENCY = 4


def _ms(dt_str: str) -> int:
    return int(datetime.strptime(dt_str, "%Y-%m-%d").replace(tzinfo=UTC).timestamp() * 1000)


async def fetch_candles(
    symbol: str,
    interval: str = "1h",
    start_date: str = "2024-11-01",
    end_date: str = "2026-05-01",
    client: httpx.AsyncClient | None = None,
) -> list[dict[str, Any]]:
    """Fetch historical candles from Binance, auto-paginating in 1000-candle chunks.

    Pass a shared ``client`` when fetching many TFs concurrently to reuse the connection pool.
    """
    start_ms = _ms(start_date)
    end_ms = _ms(end_date)
    all_candles: list[dict[str, Any]] = []

    async def _fetch(cli: httpx.AsyncClient) -> None:
        current_ms = start_ms
        while current_ms < end_ms:
            params = {
                "symbol": symbol,
                "interval": interval,
                "startTime": current_ms,
                "endTime": end_ms,
                "limit": _PAGE_LIMIT,
            }
            for attempt in range(3):
                try:
                    resp = await cli.get(BINANCE_KLINES_URL, params=params)
                    resp.raise_for_status()
                    break
                except (httpx.HTTPError, httpx.TimeoutException):
                    if attempt == 2:
                        raise
                    await asyncio.sleep(2**attempt)

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
            current_ms = raw[-1][0] + 1
            await asyncio.sleep(0.05)

    if client is not None:
        await _fetch(client)
    else:
        async with httpx.AsyncClient(timeout=30) as cli:
            await _fetch(cli)

    return all_candles


async def fetch_multi_tf_candles(
    symbol: str,
    timeframes: list[str] = DEFAULT_TIMEFRAMES,
    start_date: str = "2024-11-01",
    end_date: str = "2026-05-01",
) -> dict[str, list[dict[str, Any]]]:
    """Fetch candles for multiple timeframes concurrently. Returns {tf: [candles]}."""
    semaphore = asyncio.Semaphore(_TF_CONCURRENCY)

    async def _fetch_with_sem(tf: str, cli: httpx.AsyncClient):
        async with semaphore:
            candles = await fetch_candles(symbol, tf, start_date, end_date, client=cli)
            return tf, candles

    async with httpx.AsyncClient(timeout=30) as cli:
        pairs = await asyncio.gather(*[_fetch_with_sem(tf, cli) for tf in timeframes])

    result = {}
    for tf, candles in pairs:
        result[tf] = candles
        print(f"  {symbol} {tf}: {len(candles)} candles")
    return result


def save_candles(candles: list[dict[str, Any]], symbol: str, interval: str) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / f"{symbol}_{interval}.json"
    with open(path, "w") as f:
        json.dump(candles, f)
    return path


def load_candles(symbol: str, interval: str, data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    path = data_dir / f"{symbol}_{interval}.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)
