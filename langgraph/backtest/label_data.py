"""Hindsight labeling — look ahead N candles to determine LONG/SHORT ground truth."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


def label_candle(
    candles: List[Dict[str, Any]],
    indicator_point: Dict[str, Any],
    candle_index: int,
    lookahead: int = 24,
    atr_multiplier: float = 1.5,
) -> Optional[Dict[str, Any]]:
    """Label a single candle using hindsight.

    Returns a labeled dict or None if the candle is ambiguous/filtered.
    """
    atr = indicator_point["atr_raw"]
    ind = indicator_point["indicators"]
    entry = ind["price"]

    if candle_index + lookahead >= len(candles):
        return None  # Not enough future data

    future = candles[candle_index + 1 : candle_index + 1 + lookahead]
    if not future:
        return None

    # Quality filters
    if ind.get("volume_ratio", 0) < 0.5:
        return None
    if ind.get("adx", 0) < 15:
        return None

    threshold_pct = (atr * atr_multiplier) / entry if entry > 0 else 0

    max_up = (max(c["high"] for c in future) - entry) / entry
    max_down = (entry - min(c["low"] for c in future)) / entry

    # Directional clarity: one side must dominate by 1.5x
    if max_up > threshold_pct and max_up > max_down * 1.5:
        bias = "LONG"
        tp = entry * (1 + max_up * 0.7)  # Take 70% of the move
        sl = entry * (1 - atr * 1.0 / entry)  # 1x ATR stop
    elif max_down > threshold_pct and max_down > max_up * 1.5:
        bias = "SHORT"
        tp = entry * (1 - max_down * 0.7)
        sl = entry * (1 + atr * 1.0 / entry)
    else:
        return None  # Ambiguous

    # R:R filter
    reward = abs(tp - entry)
    risk = abs(entry - sl)
    if risk == 0 or reward / risk < 1.0:
        return None

    # Whipsaw filter: check if direction reverses within first 4 candles
    early = future[:4]
    if bias == "LONG":
        early_drop = (entry - min(c["low"] for c in early)) / entry
        if early_drop > threshold_pct:
            return None  # Whipsaw
    else:
        early_rise = (max(c["high"] for c in early) - entry) / entry
        if early_rise > threshold_pct:
            return None

    # Drawdown-before-profit check
    if bias == "LONG":
        for c in future:
            if c["low"] <= sl:
                return None  # SL hit before TP
            if c["high"] >= tp:
                break
    else:
        for c in future:
            if c["high"] >= sl:
                return None
            if c["low"] <= tp:
                break

    confidence = min(10, int((reward / risk) * 3 + ind.get("adx", 0) / 10))

    return {
        "timestamp": indicator_point["timestamp"],
        "indicators": ind,
        "atr_raw": atr,
        "label": {
            "bias": bias,
            "entry": round(entry, 2),
            "tp": round(tp, 2),
            "sl": round(sl, 2),
            "confidence": confidence,
            "quality": "HIGH" if reward / risk >= 2.0 else "MEDIUM",
        },
    }


def generate_labeled_dataset(
    candles: List[Dict[str, Any]],
    indicator_points: List[Dict[str, Any]],
    symbol: str,
    lookahead: int = 24,
) -> List[Dict[str, Any]]:
    """Generate labeled dataset from candles and indicator points."""
    # Build timestamp -> candle index map
    ts_to_idx = {c["timestamp"]: i for i, c in enumerate(candles)}

    labeled = []
    for point in indicator_points:
        idx = ts_to_idx.get(point["timestamp"])
        if idx is None:
            continue
        result = label_candle(candles, point, idx, lookahead=lookahead)
        if result is not None:
            result["symbol"] = symbol
            labeled.append(result)

    return labeled


def save_labeled_dataset(data: List[Dict[str, Any]], path: Path) -> Path:
    """Save labeled dataset as JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
    return path
