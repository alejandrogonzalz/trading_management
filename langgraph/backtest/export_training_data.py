"""Export labeled dataset as chat-format JSONL for fine-tuning.

Produces system/user/assistant messages matching generator_node() in graph.py exactly.
Splits temporally (not randomly) into train/val/test sets.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Any, List, Literal


def _build_system_prompt(mode: Literal["SPOT", "FUTURES"]) -> str:
    """Exact system prompt from generator_node()."""
    mode_context = "SPOT (Long Only, No Leverage)" if mode == "SPOT" else "FUTURES (Long/Short, Leverage 1-50x)"
    return (
        f"You are a Senior Technical Analyst for a {mode_context} trading system.\n"
        "Analyze the provided multi-timeframe indicators and generate a high-confluence trade setup.\n\n"
        f"RULES:\n- MODE: {mode}\n"
        + (
            "- If SPOT: Bias MUST be LONG. Leverage MUST be null.\n"
            if mode == "SPOT"
            else "- If FUTURES: Bias can be LONG or SHORT. Recommend leverage (1-50) based on volatility.\n"
        )
        + "- Output MUST be valid JSON."
    )


def _build_user_prompt(symbol: str, indicators: Dict[str, Any]) -> str:
    """Exact user prompt from generator_node()."""
    return (
        f"Symbol: {symbol}\n"
        f"Indicators: {json.dumps(indicators)}\n\n"
        "Return valid JSON with these exact fields:\n"
        '{\n    "bias": "LONG" or "SHORT",\n'
        '    "entry": float (price number),\n'
        '    "tp": float (take profit price),\n'
        '    "sl": float (stop loss price), \n'
        '    "leverage": integer or null,\n'
        '    "reasoning": "2 sentences max explaining the setup",\n'
        '    "quality": "HIGH", "MEDIUM", or "LOW"\n}\n\n'
        "Example JSON response:\n"
        '{\n    "bias": "LONG",\n    "entry": 100.50,\n    "tp": 105.25,\n'
        '    "sl": 98.75,\n    "leverage": 5,\n'
        '    "reasoning": "Bullish breakout on 1h with strong volume support.",\n'
        '    "quality": "HIGH"\n}'
    )


def _generate_reasoning(bias: str, indicators: Dict[str, Any]) -> str:
    """Programmatic reasoning from indicator signals."""
    parts = []
    rsi = indicators.get("rsi", 50)
    adx = indicators.get("adx", 20)
    structure = indicators.get("structure", "")
    heatmap = indicators.get("heatmap", "")
    volume_ratio = indicators.get("volume_ratio", 1.0)

    if bias == "LONG":
        if rsi < 35:
            parts.append(f"RSI at {rsi:.0f} indicates oversold conditions")
        if "BULLISH" in heatmap.upper():
            parts.append(f"{heatmap.lower().replace('_', ' ')} heatmap confirms upward momentum")
        if structure == "BREAKOUT":
            parts.append("price breaking above recent resistance")
    else:
        if rsi > 65:
            parts.append(f"RSI at {rsi:.0f} indicates overbought conditions")
        if "BEARISH" in heatmap.upper():
            parts.append(f"{heatmap.lower().replace('_', ' ')} heatmap confirms downward pressure")
        if structure == "BREAKDOWN":
            parts.append("price breaking below recent support")

    if adx > 25:
        parts.append(f"ADX at {adx:.0f} confirms strong trend")
    if volume_ratio > 1.5:
        parts.append(f"volume {volume_ratio:.1f}x above average supports the move")

    return (
        ". ".join(parts[:3]) + "."
        if parts
        else f"{'Bullish' if bias == 'LONG' else 'Bearish'} setup based on technical confluence."
    )


def build_training_example(sample: Dict[str, Any], mode: Literal["SPOT", "FUTURES"] = "FUTURES") -> Dict[str, Any]:
    """Convert one labeled sample to chat-format training example."""
    symbol = sample.get("symbol", "BTCUSDT")
    indicators = sample["indicators"]
    label = sample["label"]

    # If indicators are already multi-TF ({"1h": {...}, "4h": {...}}), use as-is.
    # If flat ({"price": ..., "rsi": ...}), wrap in {"1h": ...} for backward compat.
    if indicators and isinstance(next(iter(indicators.values())), dict):
        multi_tf_indicators = indicators
        base_ind = indicators.get("1h", next(iter(indicators.values())))
    else:
        multi_tf_indicators = {"1h": indicators}
        base_ind = indicators

    system_prompt = _build_system_prompt(mode)
    user_prompt = _build_user_prompt(symbol, multi_tf_indicators)

    # Build assistant response
    reasoning = _generate_reasoning(label["bias"], base_ind)
    assistant_response = {
        "bias": label["bias"],
        "entry": label["entry"],
        "tp": label["tp"],
        "sl": label["sl"],
        "leverage": None if mode == "SPOT" else 5,
        "reasoning": reasoning,
        "quality": label.get("quality", "MEDIUM"),
    }

    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": json.dumps(assistant_response)},
        ]
    }


def temporal_split(
    samples: List[Dict[str, Any]],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[List[Dict], List[Dict], List[Dict]]:
    """Split samples temporally (by timestamp). Assumes samples are sorted."""
    sorted_samples = sorted(samples, key=lambda s: s.get("timestamp", 0))
    n = len(sorted_samples)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return sorted_samples[:train_end], sorted_samples[train_end:val_end], sorted_samples[val_end:]


def print_stats(samples: List[Dict], split_name: str) -> None:
    """Print dataset statistics for a split."""
    if not samples:
        print(f"  {split_name}: 0 samples")
        return

    biases = Counter(s["label"]["bias"] for s in samples)
    qualities = Counter(s["label"].get("quality", "?") for s in samples)
    symbols = Counter(s.get("symbol", "?") for s in samples)

    total = len(samples)
    print(f"  {split_name}: {total} samples")
    print(f"    Bias:    {', '.join(f'{k}: {v} ({v / total * 100:.0f}%)' for k, v in biases.most_common())}")
    print(f"    Quality: {', '.join(f'{k}: {v}' for k, v in qualities.most_common())}")
    print(f"    Symbols: {', '.join(f'{k}: {v}' for k, v in symbols.most_common())}")


def export_training_data(
    dataset_path: str,
    output_dir: str,
    mode: Literal["SPOT", "FUTURES"] = "FUTURES",
) -> Dict[str, int]:
    """Main export: read labeled JSONL, build chat examples, split, write files."""
    samples = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not samples:
        print("No samples found in dataset.")
        return {"train": 0, "val": 0, "test": 0}

    train, val, test = temporal_split(samples)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    counts = {}
    for split_name, split_data in [("train", train), ("val", val), ("test", test)]:
        path = out / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for sample in split_data:
                example = build_training_example(sample, mode=mode)
                f.write(json.dumps(example) + "\n")
        counts[split_name] = len(split_data)

    # Print statistics
    print(f"\nDataset exported to {out}/")
    print(f"Total: {len(samples)} samples\n")
    print_stats(train, "Train")
    print_stats(val, "Val")
    print_stats(test, "Test")

    return counts
