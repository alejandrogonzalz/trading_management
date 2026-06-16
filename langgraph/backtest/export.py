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


def _atr_based_tp_sl(
    entry: float, bias: str, atr: float, tp_mult: float = 1.5, sl_mult: float = 1.0
) -> tuple[float, float]:
    """Compute TP/SL from ATR only — no hindsight future prices.

    This is the forward-looking formula knowable at decision time:
      LONG: TP = entry + atr * tp_mult,  SL = entry - atr * sl_mult
      SHORT: TP = entry - atr * tp_mult, SL = entry + atr * sl_mult

    Matches the production formula in simulate_trade_atr() and the LangGraph
    evaluator_node() guardrails. The model learns to output these values,
    breaking the circular dependency where training labels contained
    hindsight-derived price targets.
    """
    if bias == "LONG":
        tp = entry + atr * tp_mult
        sl = entry - atr * sl_mult
    else:
        tp = entry - atr * tp_mult
        sl = entry + atr * sl_mult
    return round(tp, 2), round(sl, 2)


def build_training_example(
    sample: dict[str, Any],
    mode: Literal["SPOT", "FUTURES"] = "FUTURES",
    use_atr_tp_sl: bool = False,
) -> dict[str, Any]:
    """Convert one labeled sample to chat-format training example.

    When ``use_atr_tp_sl=True``, replaces the hindsight-derived TP/SL from
    the label with ATR-based forward-looking values. This breaks the circular
    dependency documented in AUDIT_QLORA_88PCT.md §6: the model no longer
    learns to replicate price targets derived from future data.
    """
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

    if use_atr_tp_sl:
        atr = sample.get("atr_raw", 0)
        tp, sl = _atr_based_tp_sl(label["entry"], label["bias"], atr)
    else:
        tp = label["tp"]
        sl = label["sl"]

    reasoning = _generate_reasoning(label["bias"], base_ind)
    assistant_response = {
        "bias": label["bias"],
        "entry": label["entry"],
        "tp": tp,
        "sl": sl,
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
    use_atr_tp_sl: bool = False,
) -> dict[str, int]:
    """Read labeled JSONL, build chat examples, split, write files.

    When ``use_atr_tp_sl=True``, training labels use forward-looking ATR-based
    TP/SL instead of hindsight-derived values. This produces a model that learns
    legitimate exit placement without circular dependency on future prices.
    """
    samples = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if not samples:
        print("No samples found in dataset.")
        return {"train": 0, "val": 0, "test": 0}

    if use_atr_tp_sl:
        print("  [ATR TP/SL mode] Training labels will use forward-looking ATR exits")
        print("    TP = entry ± 1.5×ATR, SL = entry ∓ 1.0×ATR (no hindsight)")

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
                example = build_training_example(sample, mode=mode, use_atr_tp_sl=use_atr_tp_sl)
                f.write(json.dumps(example) + "\n")
        counts[split_name] = len(split_data)

    print(f"\nDataset exported to {out}/")
    print(f"Total: {len(samples)} samples")
    if use_atr_tp_sl:
        print("  Mode: ATR-based TP/SL (forward-looking, no label leakage)\n")
    else:
        print("  Mode: Hindsight TP/SL (original, circular dependency)\n")
    print_stats(train, "Train")
    print_stats(val, "Val")
    print_stats(test, "Test")

    return counts
