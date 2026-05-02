"""Core backtest runner — feed labeled data to LLM and simulate trades."""

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent dir so we can import agent modules
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm_factory import get_llm_provider
from backtest.metrics import compute_all_metrics

RESULTS_DIR = Path(__file__).parent / "data" / "results"


def _build_prompts(sample: Dict[str, Any]) -> tuple[str, str]:
    """Build system + user prompts matching generator_node format exactly."""
    symbol = sample.get("symbol", "BTCUSDT")
    indicators = sample["indicators"]

    # If indicators are already multi-TF ({"1h": {...}, "4h": {...}}), use as-is.
    # If flat ({"price": ..., "rsi": ...}), wrap in {"1h": ...} for backward compat.
    if indicators and isinstance(next(iter(indicators.values())), dict):
        multi_tf = indicators
    else:
        multi_tf = {"1h": indicators}

    system_prompt = """
    You are a Senior Technical Analyst for a FUTURES (Long/Short, Leverage 1-50x) trading system.
    Analyze the provided multi-timeframe indicators and generate a high-confluence trade setup.
    
    RULES:
    - MODE: FUTURES
    - Bias can be LONG or SHORT. Recommend leverage (1-50) based on volatility.
    - Output MUST be valid JSON.
    """

    user_prompt = f"""
    Symbol: {symbol}
    Indicators: {json.dumps(multi_tf)}
    
    Return valid JSON with these exact fields:
    {{
        "bias": "LONG" or "SHORT",
        "entry": float (price number),
        "tp": float (take profit price),
        "sl": float (stop loss price), 
        "leverage": integer or null,
        "reasoning": "2 sentences max explaining the setup",
        "quality": "HIGH", "MEDIUM", or "LOW",
        "confidence": integer 1-10
    }}
    """
    return system_prompt, user_prompt


def _parse_prediction(response_text: str) -> Optional[Dict[str, Any]]:
    """Parse LLM response into a prediction dict."""
    text = response_text.strip().replace("```json", "").replace("```", "").strip()
    import re

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        # Validate required fields
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
        else:  # SHORT
            if candle["high"] >= sl:
                return {"outcome": "LOSS", "pnl_pct": (entry - sl) / entry * 100, "hold_bars": i + 1}
            if candle["low"] <= tp:
                return {"outcome": "WIN", "pnl_pct": (entry - tp) / entry * 100, "hold_bars": i + 1}

    # Timeout
    if future_candles:
        last = future_candles[min(max_hold - 1, len(future_candles) - 1)]["close"]
        pnl = ((last - entry) / entry * 100) if bias == "LONG" else ((entry - last) / entry * 100)
    else:
        pnl = 0
    return {"outcome": "TIMEOUT", "pnl_pct": pnl, "hold_bars": min(max_hold, len(future_candles))}


def _truncate(s: str, max_len: int = 150) -> str:
    """Truncate string with ellipsis."""
    return s if len(s) <= max_len else s[:max_len] + "…"


async def run_backtest(
    dataset_path: str,
    candles_dir: str,
    tag: str = "backtest",
    max_samples: Optional[int] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Run a full backtest: load data, call LLM, simulate trades, compute metrics."""
    # Override provider/model if specified
    if provider:
        os.environ["LLM_PROVIDER"] = provider
    if model:
        os.environ["LLM_MODEL"] = model

    llm = get_llm_provider()

    # Load labeled dataset (JSONL)
    samples = []
    with open(dataset_path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    if max_samples:
        samples = samples[:max_samples]

    print(f"Running backtest: {len(samples)} samples, provider={os.getenv('LLM_PROVIDER', 'unknown')}")

    # Load candles for trade simulation (need future candles after each sample)
    # Build a map of symbol -> candles
    candles_map: Dict[str, List[Dict]] = {}
    candles_path = Path(candles_dir)
    for f in candles_path.glob("*.json"):
        symbol = f.stem.split("_")[0]
        with open(f) as fh:
            candles_map[symbol] = json.load(fh)

    # Build timestamp -> index maps per symbol
    ts_idx_map: Dict[str, Dict[int, int]] = {}
    for sym, clist in candles_map.items():
        ts_idx_map[sym] = {c["timestamp"]: i for i, c in enumerate(clist)}

    predictions = []
    actuals = []
    trade_results = []
    errors = 0
    start_time = time.time()

    for i, sample in enumerate(samples):
        symbol = sample.get("symbol", "BTCUSDT")
        label = sample["label"]

        sys_prompt, user_prompt = _build_prompts(sample)

        # Print system prompt once
        if verbose and i == 0:
            print(f"\n📋 SYSTEM PROMPT:\n{sys_prompt.strip()}\n")

        # Call LLM with retry
        prediction = None
        for attempt in range(2):
            try:
                response = await llm.generate_setup(sys_prompt, user_prompt)
                prediction = _parse_prediction(response)
                if prediction:
                    break
            except Exception as e:
                print(f"  Sample {i + 1}: LLM error (attempt {attempt + 1}): {e}")
                await asyncio.sleep(1)

        if not prediction:
            errors += 1
            print(f"  Sample {i + 1}: Failed to get prediction, skipping")
            continue

        # Add confidence if missing
        if "confidence" not in prediction:
            prediction["confidence"] = 5

        predictions.append(prediction)
        actuals.append(label)

        # Simulate trade using actual future candles
        sym_candles = candles_map.get(symbol, [])
        sym_ts_idx = ts_idx_map.get(symbol, {})
        candle_idx = sym_ts_idx.get(sample["timestamp"])

        if candle_idx is not None and candle_idx + 1 < len(sym_candles):
            future = sym_candles[candle_idx + 1 : candle_idx + 25]
            result = simulate_trade(prediction, future)
        else:
            result = {"outcome": "ERROR", "pnl_pct": 0, "hold_bars": 0}

        trade_results.append(result)

        if verbose:
            ts = sample.get("timestamp", "")
            ts_str = f" @ {datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M')}" if ts else ""
            print(f"─── Sample {i + 1}/{len(samples)} ── {symbol}{ts_str} ───")
            print(f"📤 PROMPT (user): {_truncate(user_prompt.strip())}")
            print(f"📥 RESPONSE: {json.dumps(prediction, default=str)}")
            label_bias = label.get("bias", "?")
            label_tp = label.get("tp", "?")
            label_sl = label.get("sl", "?")
            print(f"🏷️  ACTUAL: {label_bias} (tp: {label_tp}, sl: {label_sl})")
            outcome = result["outcome"]
            pnl = result["pnl_pct"]
            icon = "✅" if outcome == "WIN" else "❌" if outcome == "LOSS" else "⏱️"
            print(f"📊 RESULT: {icon} {outcome} ({pnl:+.2f}%)\n")

        # Progress
        if (i + 1) % 10 == 0 or i == len(samples) - 1:
            elapsed = time.time() - start_time
            print(f"  Progress: {i + 1}/{len(samples)} ({elapsed:.1f}s, {errors} errors)")

    elapsed = time.time() - start_time

    # Compute metrics
    metrics = compute_all_metrics(predictions, actuals, trade_results) if predictions else {}

    result = {
        "tag": tag,
        "provider": os.getenv("LLM_PROVIDER", "unknown"),
        "model": os.getenv("LLM_MODEL", "unknown"),
        "dataset": str(dataset_path),
        "total_samples": len(samples),
        "successful_predictions": len(predictions),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 2),
        "metrics": metrics,
        "predictions": predictions,
        "actuals": actuals,
        "trade_results": trade_results,
    }

    # Save results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return result
