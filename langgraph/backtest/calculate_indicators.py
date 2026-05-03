"""Batch indicator calculation over historical candles, reusing indicator_service logic."""

import math

import pandas as pd
import talib
from typing import List, Dict, Any, Optional


def _safe_float(v: float, default: float = 0.0) -> float:
    """Convert to float, replacing NaN/Inf with default."""
    f = float(v)
    return default if (math.isnan(f) or math.isinf(f)) else f


def calculate_indicators_batch(candles: List[Dict[str, Any]], lookback: int = 200) -> List[Dict[str, Any]]:
    """Calculate indicators for each candle where enough history exists.

    For each candle at index i (where i >= lookback), computes indicators
    using candles[0:i+1] — matching the indicator_service.py logic exactly.

    Returns list of dicts with {timestamp, symbol, indicators, atr_raw}.
    """
    if len(candles) < lookback + 1:
        raise ValueError(f"Need at least {lookback + 1} candles, got {len(candles)}")

    df = pd.DataFrame(candles)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    # Pre-compute full-length indicator arrays (much faster than per-candle)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values

    ema20 = talib.EMA(close, timeperiod=20)
    ema50 = talib.EMA(close, timeperiod=50)
    ema200 = talib.EMA(close, timeperiod=200)
    adx_arr = talib.ADX(high, low, close, timeperiod=14)
    rsi_arr = talib.RSI(close, timeperiod=14)
    macd_, macd_sig_, macd_hist_ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    atr_arr = talib.ATR(high, low, close, timeperiod=14)
    vol_sma20 = talib.SMA(volume, timeperiod=20)
    atr_sma20 = talib.SMA(atr_arr, timeperiod=20)
    bb_upper, bb_mid, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

    results: List[Dict[str, Any]] = []

    for i in range(lookback, len(df)):
        price = close[i]
        e20, e50, e200 = ema20[i], ema50[i], ema200[i]
        strength = adx_arr[i]

        # Heatmap
        if price > e20 > e50 > e200:
            heatmap = "STRONG_BULLISH" if strength > 25 else "BULLISH"
        elif price < e20 < e50 < e200:
            heatmap = "STRONG_BEARISH" if strength > 25 else "BEARISH"
        else:
            heatmap = "NEUTRAL"

        # Structure
        struct_lookback = 20
        start_idx = max(0, i - struct_lookback)
        prev_highs = high[start_idx:i]
        if len(prev_highs) > 0 and price > prev_highs.max():
            structure = "BREAKOUT"
        else:
            last5_close = close[max(0, i - 4) : i + 1]
            if len(last5_close) >= 3 and last5_close[-1] > last5_close[-3]:
                structure = "BULLISH"
            elif len(last5_close) >= 3 and last5_close[-1] < last5_close[-3]:
                structure = "BEARISH"
            else:
                structure = "RANGE"

        # Volume ratio
        vol_ratio = volume[i] / vol_sma20[i] if vol_sma20[i] > 0 else 1.0

        # ATR ratio
        atr_ratio = atr_arr[i] / atr_sma20[i] if atr_sma20[i] > 0 else 1.0

        # BB position
        u, bb_low = bb_upper[i], bb_lower[i]
        bb_pos = (price - bb_low) / (u - bb_low) if (u - bb_low) != 0 else 0.5

        results.append(
            {
                "timestamp": int(df.iloc[i]["timestamp"]),
                "indicators": {
                    "price": float(price),
                    "heatmap": heatmap,
                    "structure": structure,
                    "rsi": _safe_float(rsi_arr[i], 50),
                    "macd_hist": _safe_float(macd_hist_[i]),
                    "adx": _safe_float(adx_arr[i], 20),
                    "volume_ratio": _safe_float(vol_ratio, 1.0),
                    "atr_ratio": _safe_float(atr_ratio, 1.0),
                    "bb_pos": _safe_float(bb_pos, 0.5),
                },
                "atr_raw": _safe_float(atr_arr[i]),
            }
        )

    return results


def calculate_multi_tf_indicators(
    candles_by_tf: Dict[str, List[Dict[str, Any]]],
    base_tf: str = "1h",
    lookback: int = 200,
) -> List[Dict[str, Any]]:
    """Calculate indicators for all timeframes, aligned to base_tf timestamps.

    For each base_tf candle, finds the most recent candle at or before that
    timestamp in each higher timeframe and includes its indicators.

    Returns list of {timestamp, indicators: {"1h": {...}, "4h": {...}, "1d": {...}}, atr_raw}.
    """
    if base_tf not in candles_by_tf:
        raise ValueError(f"Base timeframe '{base_tf}' not in candles_by_tf")

    # Calculate indicators for each timeframe independently
    # Higher TFs have fewer candles, so use a reduced lookback to get more output points.
    # EMA200 will be NaN for early candles but RSI/MACD/ADX/BB still work fine.
    HTF_LOOKBACK = 50  # Enough for RSI-14, MACD-26, ADX-14, BB-20
    ind_by_tf: Dict[str, List[Dict[str, Any]]] = {}
    for tf, candles in candles_by_tf.items():
        n = len(candles)
        tf_lookback = lookback if tf == base_tf else min(HTF_LOOKBACK, n - 1)
        if tf_lookback < 30:
            print(f"  Warning: {tf} has only {n} candles (need >=31), skipping")
            continue
        try:
            ind_by_tf[tf] = calculate_indicators_batch(candles, tf_lookback)
        except ValueError as e:
            print(f"  Warning: {tf} indicator calculation failed: {e}, skipping")
            continue

    if base_tf not in ind_by_tf:
        raise ValueError(f"Not enough candles for base timeframe '{base_tf}'")

    # Build timestamp -> indicators lookup for higher TFs
    # For each higher TF, sort by timestamp so we can binary-search
    htf_sorted: Dict[str, List[Dict[str, Any]]] = {}
    for tf, points in ind_by_tf.items():
        if tf != base_tf:
            htf_sorted[tf] = sorted(points, key=lambda p: p["timestamp"])

    def _find_latest_at_or_before(sorted_points: List[Dict[str, Any]], ts: int) -> Optional[Dict[str, Any]]:
        """Binary search for the most recent point at or before ts."""
        lo, hi, best = 0, len(sorted_points) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_points[mid]["timestamp"] <= ts:
                best = sorted_points[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    # Align: for each base_tf point, attach the latest indicator from each higher TF
    results: List[Dict[str, Any]] = []
    for point in ind_by_tf[base_tf]:
        multi_ind: Dict[str, Any] = {base_tf: point["indicators"]}

        for tf, sorted_points in htf_sorted.items():
            match = _find_latest_at_or_before(sorted_points, point["timestamp"])
            if match:
                multi_ind[tf] = match["indicators"]

        results.append(
            {
                "timestamp": point["timestamp"],
                "indicators": multi_ind,
                "atr_raw": point["atr_raw"],
            }
        )

    return results
