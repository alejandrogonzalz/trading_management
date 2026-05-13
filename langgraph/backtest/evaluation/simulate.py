"""Trade simulation and LLM response parsing."""

import json
import re
from typing import Dict, List, Any, Optional


def _parse_prediction(response_text: str) -> Optional[Dict[str, Any]]:
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


def simulate_trade(prediction: Dict, future_candles: List[Dict], max_hold: int = 24) -> Dict[str, Any]:
    """Simulate a single trade against future candles."""
    entry = prediction.get("entry", 0)
    tp = prediction.get("tp", 0)
    sl = prediction.get("sl", 0)
    bias = prediction.get("bias", "LONG")

    if not entry or not tp or not sl:
        return {"outcome": "ERROR", "pnl_pct": 0, "hold_bars": 0}

    for i, candle in enumerate(future_candles[:max_hold]):
        if bias == "LONG":
            if candle["low"] <= sl:
                return {"outcome": "LOSS", "pnl_pct": (sl - entry) / entry * 100, "hold_bars": i + 1}
            if candle["high"] >= tp:
                return {"outcome": "WIN", "pnl_pct": (tp - entry) / entry * 100, "hold_bars": i + 1}
        else:
            if candle["high"] >= sl:
                return {"outcome": "LOSS", "pnl_pct": (entry - sl) / entry * 100, "hold_bars": i + 1}
            if candle["low"] <= tp:
                return {"outcome": "WIN", "pnl_pct": (entry - tp) / entry * 100, "hold_bars": i + 1}

    if future_candles:
        last = future_candles[min(max_hold - 1, len(future_candles) - 1)]["close"]
        pnl = ((last - entry) / entry * 100) if bias == "LONG" else ((entry - last) / entry * 100)
    else:
        pnl = 0
    return {"outcome": "TIMEOUT", "pnl_pct": pnl, "hold_bars": min(max_hold, len(future_candles))}
