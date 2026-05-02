"""Batch indicator calculation over historical candles, reusing indicator_service logic."""

import pandas as pd
import talib
from typing import List, Dict, Any


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

        results.append({
            "timestamp": int(df.iloc[i]["timestamp"]),
            "indicators": {
                "price": float(price),
                "heatmap": heatmap,
                "structure": structure,
                "rsi": float(rsi_arr[i]),
                "macd_hist": float(macd_hist_[i]),
                "adx": float(adx_arr[i]),
                "volume_ratio": float(vol_ratio),
                "atr_ratio": float(atr_ratio),
                "bb_pos": float(bb_pos),
            },
            "atr_raw": float(atr_arr[i]),
        })

    return results
