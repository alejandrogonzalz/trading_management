"""Export labeled dataset as chat-format JSONL for fine-tuning.

Produces system/user/assistant messages matching generator_node() in graph.py exactly.
Splits temporally (not randomly) into train/val/test sets.
"""

import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from agent.prompts import build_system_prompt, build_user_prompt
from backtest.models.features import _temporal_split


def _generate_reasoning(bias: str, indicators: dict[str, Any]) -> str:
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


def build_training_example(sample: dict[str, Any], mode: Literal["SPOT", "FUTURES"] = "FUTURES") -> dict[str, Any]:
    """Convert one labeled sample to chat-format training example."""
    symbol = sample.get("symbol", "BTCUSDT")
    indicators = sample["indicators"]
    label = sample["label"]

    if indicators and isinstance(next(iter(indicators.values())), dict):
        multi_tf_indicators = indicators
        base_ind = indicators.get("1h", next(iter(indicators.values())))
    else:
        multi_tf_indicators = {"1h": indicators}
        base_ind = indicators

    system_prompt = build_system_prompt(mode)
    user_prompt = build_user_prompt(symbol, multi_tf_indicators)

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


def print_stats(samples: list[dict], split_name: str) -> None:
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
) -> dict[str, int]:
    """Read labeled JSONL, build chat examples, split, write files."""
    samples = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not samples:
        print("No samples found in dataset.")
        return {"train": 0, "val": 0, "test": 0}

    # Use the SAME split function as the ML/LLM runners so the fine-tuning
    # train/val/test partitions are identical to those of LSTM/XGBoost. That
    # function now does a strict temporal holdout (global sort by timestamp +
    # lookahead embargo), which is what keeps the paired McNemar / t-test
    # comparison across models valid AND leak-free.
    train, val, test = _temporal_split(samples)

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

    print(f"\nDataset exported to {out}/")
    print(f"Total: {len(samples)} samples\n")
    print_stats(train, "Train")
    print_stats(val, "Val")
    print_stats(test, "Test")

    return counts
