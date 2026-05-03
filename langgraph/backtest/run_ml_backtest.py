"""Backtest runner for traditional ML models — same output format as LLM backtest."""

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from backtest.metrics import compute_all_metrics
from backtest.ml_models import get_predictor, _load_dataset, _temporal_split
from backtest.run_backtest import simulate_trade

RESULTS_DIR = Path(__file__).parent / "data" / "results"


def run_ml_backtest(
    dataset_path: str,
    model_type: str = "xgboost",
    tag: Optional[str] = None,
    max_samples: Optional[int] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Train ML model and backtest on the test split.

    Uses temporal split: train 70%, val 15%, test 15%.
    Output format matches LLM backtest for direct comparison via compare.py.
    """
    tag = tag or f"ml-{model_type}"
    candles_dir = Path(__file__).parent / "data" / "candles"

    # Load candles for trade simulation (1h base TF only — matches labeled data timestamps)
    candles_map: Dict[str, list] = {}
    for f in candles_dir.glob("*_1h.json"):
        symbol = f.stem.split("_")[0]
        with open(f) as fh:
            candles_map[symbol] = json.load(fh)
    # Fallback: load any candle file if no _1h found
    if not candles_map:
        for f in candles_dir.glob("*.json"):
            symbol = f.stem.split("_")[0]
            if symbol not in candles_map:
                with open(f) as fh:
                    candles_map[symbol] = json.load(fh)

    ts_idx_map: Dict[str, Dict[int, int]] = {}
    for sym, clist in candles_map.items():
        ts_idx_map[sym] = {c["timestamp"]: i for i, c in enumerate(clist)}

    # Train
    predictor = get_predictor(model_type)
    print(f"Training {model_type}...")
    start_time = time.time()
    train_info = predictor.train(dataset_path)
    train_elapsed = time.time() - start_time
    print(
        f"  Train: {train_info['train_size']} samples, Val accuracy: {train_info['val_accuracy']:.3f} ({train_elapsed:.1f}s)"
    )

    # Get test split
    all_samples = _load_dataset(dataset_path)
    _train, _val, test = _temporal_split(all_samples)
    if max_samples:
        test = test[:max_samples]
    print(f"Testing on {len(test)} samples...")

    # Predict and simulate
    predictions = []
    actuals = []
    trade_results = []
    errors = 0

    for i, sample in enumerate(test):
        symbol = sample.get("symbol", "BTCUSDT")
        label = sample["label"]
        atr = sample.get("atr_raw", 0)

        pred = predictor.predict(sample["indicators"])

        # Generate TP/SL from ATR (ML only predicts direction)
        entry = label["entry"]  # Use actual entry price
        if atr > 0:
            if pred["bias"] == "LONG":
                tp = entry + 1.5 * atr
                sl = entry - 1.0 * atr
            else:
                tp = entry - 1.5 * atr
                sl = entry + 1.0 * atr
        else:
            # Fallback: use label's TP/SL distances
            tp = label["tp"]
            sl = label["sl"]

        prediction = {
            "bias": pred["bias"],
            "entry": entry,
            "tp": round(tp, 2),
            "sl": round(sl, 2),
            "confidence": pred["confidence"],
            "quality": "MEDIUM",
        }

        predictions.append(prediction)
        actuals.append(label)

        # Simulate trade
        sym_candles = candles_map.get(symbol, [])
        sym_ts_idx = ts_idx_map.get(symbol, {})
        candle_idx = sym_ts_idx.get(sample["timestamp"])

        if candle_idx is not None and candle_idx + 1 < len(sym_candles):
            future = sym_candles[candle_idx + 1 : candle_idx + 25]
            result = simulate_trade(prediction, future)
        else:
            result = {"outcome": "ERROR", "pnl_pct": 0, "hold_bars": 0}
            errors += 1

        trade_results.append(result)

        if verbose:
            ts = sample.get("timestamp", "")
            ts_str = f" @ {datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M')}" if ts else ""
            outcome = result["outcome"]
            pnl = result["pnl_pct"]
            icon = "✅" if outcome == "WIN" else "❌" if outcome == "LOSS" else "⏱️"
            match = "✓" if pred["bias"] == label["bias"] else "✗"
            print(
                f"  {i + 1}/{len(test)} {symbol}{ts_str} pred={pred['bias']} actual={label['bias']} {match} | {icon} {outcome} ({pnl:+.2f}%)"
            )

    elapsed = time.time() - start_time
    metrics = compute_all_metrics(predictions, actuals, trade_results) if predictions else {}

    result = {
        "tag": tag,
        "provider": model_type,
        "model": model_type,
        "dataset": str(dataset_path),
        "total_samples": len(test),
        "successful_predictions": len(predictions),
        "errors": errors,
        "elapsed_seconds": round(elapsed, 2),
        "train_info": train_info,
        "metrics": metrics,
        "predictions": predictions,
        "actuals": actuals,
        "trade_results": trade_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    return result
