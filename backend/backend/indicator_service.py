import pandas as pd
import numpy as np
import talib
from typing import List, Dict, Any, Optional

def calculate_indicators(candle_data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Calculates technical indicators strictly following table_definitions.txt
    """
    if not candle_data or len(candle_data) < 200:
        return {"error": "Not enough candle data (need 200+ candles)."}

    df = pd.DataFrame(candle_data)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # 1. Price
    current_price = df['close'].iloc[-1]

    # 2. Heatmap (EMA Alignment + Price Confirmation + Trend Strength)
    ema20 = talib.EMA(df['close'], timeperiod=20)
    ema50 = talib.EMA(df['close'], timeperiod=50)
    ema200 = talib.EMA(df['close'], timeperiod=200)
    adx_series = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14)
    
    e20, e50, e200 = ema20.iloc[-1], ema50.iloc[-1], ema200.iloc[-1]
    strength = adx_series.iloc[-1]
    
    if current_price > e20 > e50 > e200:
        heatmap = "STRONG_BULLISH" if strength > 25 else "BULLISH"
    elif current_price < e20 < e50 < e200:
        heatmap = "STRONG_BEARISH" if strength > 25 else "BEARISH"
    else:
        heatmap = "NEUTRAL"

    # 3. Structure (Market Structure)
    # recent_high = max(high last N candles) - let's use 20 for 'recent'
    lookback = 20
    prev_highs = df['high'].iloc[-(lookback+1):-1]
    recent_max = prev_highs.max()
    
    if current_price > recent_max:
        structure = "BREAKOUT"
    else:
        # Check for HH/HL vs LL/LH
        last_5 = df.iloc[-5:]
        if (last_5['high'].is_monotonic_increasing or last_5['close'].iloc[-1] > last_5['close'].iloc[-3]):
            structure = "BULLISH"
        elif (last_5['low'].is_monotonic_decreasing or last_5['close'].iloc[-1] < last_5['close'].iloc[-3]):
            structure = "BEARISH"
        else:
            structure = "RANGE"

    # 4. RSI
    rsi = talib.RSI(df['close'], timeperiod=14).iloc[-1]

    # 5. MACD (Histogram)
    macd, macdsignal, macdhist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    macd_hist = macdhist.iloc[-1]

    # 6. ADX
    adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]

    # 7. Volume Ratio (current_volume / avg_volume_20)
    vol_sma20 = talib.SMA(df['volume'], timeperiod=20)
    volume_ratio = df['volume'].iloc[-1] / vol_sma20.iloc[-1] if vol_sma20.iloc[-1] > 0 else 1.0

    # 8. ATR Ratio (ATR_current / avg_ATR_20)
    atr_series = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    avg_atr_20 = talib.SMA(atr_series, timeperiod=20).iloc[-1]
    atr_ratio = atr_series.iloc[-1] / avg_atr_20 if avg_atr_20 > 0 else 1.0

    # 9. BB Position ((price - lower) / (upper - lower))
    upper, middle, lower = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    u, l = upper.iloc[-1], lower.iloc[-1]
    bb_pos = (current_price - l) / (u - l) if (u - l) != 0 else 0.5

    return {
        "price": float(current_price),
        "heatmap": heatmap,
        "structure": structure,
        "rsi": float(rsi),
        "macd_hist": float(macd_hist),
        "adx": float(adx),
        "volume_ratio": float(volume_ratio),
        "atr_ratio": float(atr_ratio),
        "bb_pos": float(bb_pos)
    }
