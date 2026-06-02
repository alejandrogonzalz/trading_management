from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.services.binance_service import binance_client  # Relative import


def get_candles(symbol: str, interval: str, limit: int = 200) -> list[dict[str, Any]]:
    """
    Fetches candlestick data from Binance.
    Adjusts limit if data is scarce (for 1w, 1M).
    """
    try:
        # If interval is 1w or 1M, we might accept fewer candles if 200 is not available
        actual_limit = limit
        klines = binance_client.get_klines(symbol=symbol, interval=interval, limit=actual_limit)

        if not klines or len(klines) < 30:  # Minimum 30 candles for basic indicators
            return []

        processed_klines = []
        for kline in klines:
            processed_klines.append(
                {
                    "time": kline[0] / 1000,
                    "open": float(kline[1]),
                    "high": float(kline[2]),
                    "low": float(kline[3]),
                    "close": float(kline[4]),
                    "volume": float(kline[5]),
                }
            )
        return processed_klines
    except Exception as e:
        print(f"Error fetching {symbol} {interval}: {e}")
        return []


def get_multi_timeframe_candles(symbol: str) -> dict[str, list[dict[str, Any]]]:
    """
    Fetches candle data for multiple timeframes IN PARALLEL.
    """
    supported_intervals = ["5m", "15m", "1h", "4h", "1d", "1w", "1M"]

    def fetch_task(tf):
        return tf, get_candles(symbol, tf)

    multi_timeframe_data = {}
    with ThreadPoolExecutor(max_workers=len(supported_intervals)) as executor:
        results = list(executor.map(fetch_task, supported_intervals))
        for tf, data in results:
            multi_timeframe_data[tf] = data

    return multi_timeframe_data
