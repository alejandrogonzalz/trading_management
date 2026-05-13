"""Batch indicator calculation over historical candles using TA-Lib."""

import math
from typing import Any, Dict, List, Optional

import pandas as pd
import talib


def _safe_float(v: float, default: float = 0.0) -> float:
    f = float(v)
    return default if (math.isnan(f) or math.isinf(f)) else f


def calculate_indicators_batch(candles: List[Dict[str, Any]], lookback: int = 200) -> List[Dict[str, Any]]:
    """Calculate indicators for each candle where enough history exists.

    For each candle at index i (where i >= lookback), computes indicators
    using candles[0:i+1] — matching the indicator_service.py logic exactly.
    Returns list of dicts with {timestamp, indicators, atr_raw}.
    """
    if len(candles) < lookback + 1:
        raise ValueError(f"Need at least {lookback + 1} candles, got {len(candles)}")

    df = pd.DataFrame(candles)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)

    close, high, low, volume = df["close"].values, df["high"].values, df["low"].values, df["volume"].values

    ema20 = talib.EMA(close, timeperiod=20)
    ema50 = talib.EMA(close, timeperiod=50)
    ema200 = talib.EMA(close, timeperiod=200)
    adx_arr = talib.ADX(high, low, close, timeperiod=14)
    rsi_arr = talib.RSI(close, timeperiod=14)
    _, _, macd_hist_ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    atr_arr = talib.ATR(high, low, close, timeperiod=14)
    vol_sma20 = talib.SMA(volume, timeperiod=20)
    atr_sma20 = talib.SMA(atr_arr, timeperiod=20)
    bb_upper, _, bb_lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)

    results = []
    for i in range(lookback, len(df)):
        price = close[i]
        e20, e50, e200 = ema20[i], ema50[i], ema200[i]
        strength = adx_arr[i]

        if price > e20 > e50 > e200:
            heatmap = "STRONG_BULLISH" if strength > 25 else "BULLISH"
        elif price < e20 < e50 < e200:
            heatmap = "STRONG_BEARISH" if strength > 25 else "BEARISH"
        else:
            heatmap = "NEUTRAL"

        start_idx = max(0, i - 20)
        prev_highs = high[start_idx:i]
        if len(prev_highs) > 0 and price > prev_highs.max():
            structure = "BREAKOUT"
        else:
            last5 = close[max(0, i - 4): i + 1]
            if len(last5) >= 3 and last5[-1] > last5[-3]:
                structure = "BULLISH"
            elif len(last5) >= 3 and last5[-1] < last5[-3]:
                structure = "BEARISH"
            else:
                structure = "RANGE"

        vol_ratio = volume[i] / vol_sma20[i] if vol_sma20[i] > 0 else 1.0
        atr_ratio = atr_arr[i] / atr_sma20[i] if atr_sma20[i] > 0 else 1.0
        u, bl = bb_upper[i], bb_lower[i]
        bb_pos = (price - bl) / (u - bl) if (u - bl) != 0 else 0.5

        results.append({
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
        })

    return results


def calculate_multi_tf_indicators(
    candles_by_tf: Dict[str, List[Dict[str, Any]]],
    base_tf: str = "1h",
    lookback: int = 200,
) -> List[Dict[str, Any]]:
    """Calculate indicators for all timeframes, aligned to base_tf timestamps.

    For each base_tf candle, finds the most recent candle at or before that
    timestamp in each other timeframe and includes its indicators.
    Returns list of {timestamp, indicators: {tf: {...}, ...}, atr_raw}.
    """
    if base_tf not in candles_by_tf:
        raise ValueError(f"Base timeframe '{base_tf}' not in candles_by_tf")

    _TF_LOOKBACK: Dict[str, int] = {
        "5m": 40, "15m": 40, "30m": 40, "4h": 40, "1d": 40, "1w": 30,
    }

    ind_by_tf: Dict[str, List[Dict[str, Any]]] = {}
    for tf, candles in candles_by_tf.items():
        n = len(candles)
        tf_lookback = lookback if tf == base_tf else min(_TF_LOOKBACK.get(tf, 40), n - 1)
        if tf_lookback < 14:
            print(f"  Warning: {tf} has only {n} candles (need >=15), skipping")
            continue
        try:
            ind_by_tf[tf] = calculate_indicators_batch(candles, tf_lookback)
        except ValueError as e:
            print(f"  Warning: {tf} indicator calculation failed: {e}, skipping")
            continue

    if base_tf not in ind_by_tf:
        raise ValueError(f"Not enough candles for base timeframe '{base_tf}'")

    htf_sorted: Dict[str, List[Dict[str, Any]]] = {
        tf: sorted(pts, key=lambda p: p["timestamp"])
        for tf, pts in ind_by_tf.items() if tf != base_tf
    }

    def _find_latest_at_or_before(sorted_points: List[Dict], ts: int) -> Optional[Dict]:
        lo, hi, best = 0, len(sorted_points) - 1, None
        while lo <= hi:
            mid = (lo + hi) // 2
            if sorted_points[mid]["timestamp"] <= ts:
                best = sorted_points[mid]
                lo = mid + 1
            else:
                hi = mid - 1
        return best

    results = []
    for point in ind_by_tf[base_tf]:
        multi_ind: Dict[str, Any] = {base_tf: point["indicators"]}
        for tf, sorted_pts in htf_sorted.items():
            match = _find_latest_at_or_before(sorted_pts, point["timestamp"])
            if match:
                multi_ind[tf] = match["indicators"]
        results.append({
            "timestamp": point["timestamp"],
            "indicators": multi_ind,
            "atr_raw": point["atr_raw"],
        })

    return results
