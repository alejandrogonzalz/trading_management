"""Shared prompt builders — single source of truth for generator_node, backtest, and fine-tuning export."""

import json
from typing import Any, Literal


def build_system_prompt(mode: Literal["SPOT", "FUTURES"] = "FUTURES") -> str:
    mode_context = "SPOT (Long Only, No Leverage)" if mode == "SPOT" else "FUTURES (Long/Short, Leverage 1-50x)"
    spot_rule = "- If SPOT: Bias MUST be LONG. Leverage MUST be null.\n" if mode == "SPOT" else ""
    futures_rule = (
        "- If FUTURES: Bias can be LONG or SHORT. Recommend leverage (1-50) based on volatility.\n"
        if mode == "FUTURES"
        else ""
    )
    return (
        f"You are a Senior Technical Analyst for a {mode_context} trading system.\n"
        "Analyze the provided multi-timeframe indicators and generate a high-confluence trade setup.\n\n"
        f"RULES:\n- MODE: {mode}\n"
        f"{spot_rule}{futures_rule}"
        "- Output MUST be valid JSON."
    )


def build_user_prompt(symbol: str, indicators: dict[str, Any]) -> str:
    return (
        f"Symbol: {symbol}\n"
        f"Indicators: {json.dumps(indicators)}\n\n"
        "Return valid JSON with these exact fields:\n"
        '{\n    "bias": "LONG" or "SHORT",\n'
        '    "entry": float (price number),\n'
        '    "tp": float (take profit price),\n'
        '    "sl": float (stop loss price),\n'
        '    "leverage": integer or null,\n'
        '    "reasoning": "2 sentences max explaining the setup",\n'
        '    "quality": "HIGH", "MEDIUM", or "LOW",\n'
        '    "confidence": integer 1-10\n}\n\n'
        "Example JSON response:\n"
        '{\n    "bias": "LONG",\n    "entry": 100.50,\n    "tp": 105.25,\n'
        '    "sl": 98.75,\n    "leverage": 5,\n'
        '    "reasoning": "Bullish breakout on 1h with strong volume support.",\n'
        '    "quality": "HIGH",\n    "confidence": 8\n}'
    )
