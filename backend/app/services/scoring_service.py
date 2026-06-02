from typing import Any


def calculate_score(indicators: dict[str, Any]) -> dict[str, Any]:
    """
    Weighted confluence scoring strictly following table_definitions.txt
    """
    if not indicators or indicators.get("error"):
        return {"score": 0, "reason": indicators.get("error", "No data")}

    points = 0

    # 1. Bullish EMA alignment (+2)
    if indicators.get("heatmap") == "BULLISH":
        points += 2
    elif indicators.get("heatmap") == "BEARISH":
        points -= 2

    # 2. RSI Oversold (+2) (Spec defines oversold as <30)
    rsi = indicators.get("rsi", 50)
    if rsi < 30:
        points += 2
    elif rsi > 70:
        points -= 2

    # 3. MACD bullish (+1)
    if indicators.get("macd_hist", 0) > 0:
        points += 1
    elif indicators.get("macd_hist", 0) < 0:
        points -= 1

    # 4. ADX strong (+1) (>25 is strong)
    if indicators.get("adx", 0) > 25:
        points += 1

    # 5. Volume spike (+1) (>1.5 is spike)
    if indicators.get("volume_ratio", 1.0) > 1.5:
        points += 1

    # 6. ATR expansion (+1) (>1.3 is expansion)
    if indicators.get("atr_ratio", 1.0) > 1.3:
        points += 1

    # 7. BB oversold (+1) (<0.2 is oversold)
    if indicators.get("bb_pos", 0.5) < 0.2:
        points += 1
    elif indicators.get("bb_pos", 0.5) > 0.8:
        points -= 1

    # Max points possible in this specific model: 2+2+1+1+1+1+1 = 9
    # Scale to 0-10: (points / 9) * 10, shifted to center at 5
    # Let's use a simpler normalization for 0-10 range
    # Shifted: -7 to +9 range. Let's just clamp it.
    normalized = ((points + 7) / 16) * 10
    final_score = max(0, min(10, normalized))

    return {"score": round(final_score, 2), "raw_points": points}
