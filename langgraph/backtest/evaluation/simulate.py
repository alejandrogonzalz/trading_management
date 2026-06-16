"""Trade simulation and LLM response parsing."""

import json
import re
from typing import Any


def _parse_prediction(response_text: str) -> dict[str, Any] | None:
    """Parse LLM response into a prediction dict."""
    text = response_text.strip().replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        if "bias" not in parsed or "entry" not in parsed:
            return None
        return parsed
    except json.JSONDecodeError:
        return None


def simulate_trade(
    prediction: dict,
    future_candles: list[dict],
    max_hold: int = 24,
    fee_pct: float = 0.001,
) -> dict[str, Any]:
    """Simulate a single trade against future candles.

    Uses model-predicted TP/SL. fee_pct is applied as a round-trip cost
    (fee_pct per side, so 2 × fee_pct deducted from every pnl_pct).
    Default: 0.1% per side (Binance taker rate) = 0.2% round-trip.
    """
    entry = prediction.get("entry", 0)
    tp = prediction.get("tp", 0)
    sl = prediction.get("sl", 0)
    bias = prediction.get("bias", "LONG")
    fee = fee_pct * 2 * 100  # round-trip in percentage points

    if not entry or not tp or not sl:
        return {"outcome": "ERROR", "pnl_pct": 0, "hold_bars": 0}

    for i, candle in enumerate(future_candles[:max_hold]):
        if bias == "LONG":
            if candle["low"] <= sl:
                return {"outcome": "LOSS", "pnl_pct": (sl - entry) / entry * 100 - fee, "hold_bars": i + 1}
            if candle["high"] >= tp:
                return {"outcome": "WIN", "pnl_pct": (tp - entry) / entry * 100 - fee, "hold_bars": i + 1}
        else:
            if candle["high"] >= sl:
                return {"outcome": "LOSS", "pnl_pct": (entry - sl) / entry * 100 - fee, "hold_bars": i + 1}
            if candle["low"] <= tp:
                return {"outcome": "WIN", "pnl_pct": (entry - tp) / entry * 100 - fee, "hold_bars": i + 1}

    if future_candles:
        last = future_candles[min(max_hold - 1, len(future_candles) - 1)]["close"]
        pnl = ((last - entry) / entry * 100) if bias == "LONG" else ((entry - last) / entry * 100)
    else:
        pnl = 0
    return {"outcome": "TIMEOUT", "pnl_pct": pnl - fee, "hold_bars": min(max_hold, len(future_candles))}


def simulate_trade_atr(
    prediction: dict,
    atr_raw: float,
    future_candles: list[dict],
    tp_mult: float = 1.5,
    sl_mult: float = 1.0,
    fee_pct: float = 0.001,
    max_hold: int = 24,
) -> dict[str, Any]:
    """Forward-looking simulation using ATR-derived TP/SL.

    Ignores the model's predicted TP/SL values entirely. Uses the candle's
    ATR (available at trade time) to set exits: TP = entry ± atr * tp_mult,
    SL = entry ∓ atr * sl_mult. This breaks the circular dependency where the
    model learns hindsight-calibrated TP/SL targets from training labels.
    """
    entry = prediction.get("entry", 0)
    bias = prediction.get("bias", "LONG")
    fee = fee_pct * 2 * 100

    if not entry or not atr_raw:
        return {"outcome": "ERROR", "pnl_pct": 0, "hold_bars": 0}

    if bias == "LONG":
        tp = entry + atr_raw * tp_mult
        sl = entry - atr_raw * sl_mult
    else:
        tp = entry - atr_raw * tp_mult
        sl = entry + atr_raw * sl_mult

    for i, candle in enumerate(future_candles[:max_hold]):
        if bias == "LONG":
            if candle["low"] <= sl:
                return {"outcome": "LOSS", "pnl_pct": (sl - entry) / entry * 100 - fee, "hold_bars": i + 1}
            if candle["high"] >= tp:
                return {"outcome": "WIN", "pnl_pct": (tp - entry) / entry * 100 - fee, "hold_bars": i + 1}
        else:
            if candle["high"] >= sl:
                return {"outcome": "LOSS", "pnl_pct": (entry - sl) / entry * 100 - fee, "hold_bars": i + 1}
            if candle["low"] <= tp:
                return {"outcome": "WIN", "pnl_pct": (entry - tp) / entry * 100 - fee, "hold_bars": i + 1}

    if future_candles:
        last = future_candles[min(max_hold - 1, len(future_candles) - 1)]["close"]
        pnl = ((last - entry) / entry * 100) if bias == "LONG" else ((entry - last) / entry * 100)
    else:
        pnl = 0
    return {"outcome": "TIMEOUT", "pnl_pct": pnl - fee, "hold_bars": min(max_hold, len(future_candles))}
