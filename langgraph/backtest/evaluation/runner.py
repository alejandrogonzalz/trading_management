"""Class-based backtest runners — LLM and ML variants with identical output format."""

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.llm_factory import get_llm_provider
from agent.prompts import build_system_prompt, build_user_prompt
from backtest.evaluation.metrics import compute_all_metrics
from backtest.evaluation.simulate import _parse_prediction, simulate_trade
from backtest.models.features import _load_dataset, _temporal_split, extract_features
from backtest.models.sklearn_models import get_predictor

RESULTS_DIR = Path(__file__).parent.parent / "data" / "results"
CANDLES_DIR = Path(__file__).parent.parent / "data" / "candles"
OPTIMIZATION_RESULTS_DIR = Path(__file__).parent.parent.parent / "optimization" / "results"


def _load_jsonl(path: str) -> list[dict[str, Any]]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _load_candles_map(candles_dir: Path) -> tuple[dict[str, list], dict[str, dict[int, int]]]:
    """Load 1h candles for all symbols (used for trade simulation timestamps)."""
    candles_map: dict[str, list] = {}
    for f in candles_dir.glob("*_1h.json"):
        sym = f.stem.split("_")[0]
        with open(f) as fh:
            candles_map[sym] = json.load(fh)
    if not candles_map:
        for f in candles_dir.glob("*.json"):
            sym = f.stem.split("_")[0]
            if sym not in candles_map:
                with open(f) as fh:
                    candles_map[sym] = json.load(fh)

    ts_idx_map = {sym: {c["timestamp"]: i for i, c in enumerate(clist)} for sym, clist in candles_map.items()}
    return candles_map, ts_idx_map


def _save_result(result: dict[str, Any], tag: str) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = RESULTS_DIR / f"{tag}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2, default=str)
    return out


class LLMBacktestRunner:
    """Run a backtest by calling an LLM for each sample in a labeled dataset.

    Usage::

        runner = LLMBacktestRunner(dataset_path="...", provider="groq", tag="zero-shot-groq")
        result = asyncio.run(runner.run())
    """

    def __init__(
        self,
        dataset_path: str,
        tag: str = "backtest",
        provider: str | None = None,
        model: str | None = None,
        max_samples: int | None = None,
        candles_dir: Path | None = None,
        mode: str = "FUTURES",
        verbose: bool = False,
    ):
        self.dataset_path = dataset_path
        self.tag = tag
        self.provider = provider
        self.model = model
        self.max_samples = max_samples
        self.candles_dir = candles_dir or CANDLES_DIR
        self.mode = mode
        self.verbose = verbose

    async def run(self) -> dict[str, Any]:
        if self.provider:
            os.environ["LLM_PROVIDER"] = self.provider
        if self.model:
            os.environ["LLM_MODEL"] = self.model

        llm = get_llm_provider()
        samples = _load_jsonl(self.dataset_path)
        if self.max_samples:
            samples = samples[: self.max_samples]

        candles_map, ts_idx_map = _load_candles_map(self.candles_dir)
        system_prompt = build_system_prompt(self.mode)

        print(f"LLM backtest: {len(samples)} samples | provider={os.getenv('LLM_PROVIDER')} | mode={self.mode}")

        predictions, actuals, trade_results, sample_keys = [], [], [], []
        errors = 0
        start = time.time()

        for i, sample in enumerate(samples):
            symbol = sample.get("symbol", "BTCUSDT")
            label = sample["label"]
            indicators = sample["indicators"]
            if indicators and not isinstance(next(iter(indicators.values())), dict):
                indicators = {"1h": indicators}

            user_prompt = build_user_prompt(symbol, indicators)

            prediction = None
            for attempt in range(2):
                try:
                    response = await llm.generate_setup(system_prompt, user_prompt)
                    prediction = _parse_prediction(response)
                    if prediction:
                        break
                except Exception as exc:
                    print(f"  [{i + 1}] LLM error (attempt {attempt + 1}): {exc}")
                    await asyncio.sleep(1)

            if not prediction:
                errors += 1
                continue

            prediction.setdefault("confidence", 5)
            predictions.append(prediction)
            actuals.append(label)
            sample_keys.append(f"{symbol}@{sample['timestamp']}")

            candle_idx = ts_idx_map.get(symbol, {}).get(sample["timestamp"])
            if candle_idx is not None and candle_idx + 1 < len(candles_map.get(symbol, [])):
                future = candles_map[symbol][candle_idx + 1 : candle_idx + 25]
                trade_result = simulate_trade(prediction, future)
            else:
                trade_result = {"outcome": "ERROR", "pnl_pct": 0, "hold_bars": 0}
            trade_results.append(trade_result)

            if self.verbose:
                ts = sample.get("timestamp", "")
                ts_str = f" @ {datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M')}" if ts else ""
                icon = {"WIN": "✅", "LOSS": "❌"}.get(trade_result["outcome"], "⏱️")
                print(
                    f"[{i + 1}/{len(samples)}] {symbol}{ts_str} pred={prediction.get('bias')} actual={label['bias']} {icon} {trade_result['pnl_pct']:+.2f}%"
                )

            if (i + 1) % 50 == 0:
                print(f"  Progress: {i + 1}/{len(samples)} ({time.time() - start:.0f}s, {errors} errors)")

        metrics = compute_all_metrics(predictions, actuals, trade_results) if predictions else {}
        result = {
            "tag": self.tag,
            "provider": os.getenv("LLM_PROVIDER", "unknown"),
            "model": os.getenv("LLM_MODEL", "unknown"),
            "mode": self.mode,
            "dataset": str(self.dataset_path),
            "total_samples": len(samples),
            "successful_predictions": len(predictions),
            "errors": errors,
            "elapsed_seconds": round(time.time() - start, 2),
            "metrics": metrics,
            "predictions": predictions,
            "actuals": actuals,
            "trade_results": trade_results,
            "sample_keys": sample_keys,
        }

        out = _save_result(result, self.tag)
        print(f"\nResults → {out}")
        return result


class MLBacktestRunner:
    """Train an ML model from scratch on the dataset and evaluate on the test split.

    Usage::

        runner = MLBacktestRunner(dataset_path="...", model_type="lstm", tag="ml-lstm")
        result = runner.run()
    """

    def __init__(
        self,
        dataset_path: str,
        model_type: str = "xgboost",
        tag: str | None = None,
        max_samples: int | None = None,
        candles_dir: Path | None = None,
        serialize: bool = False,
        verbose: bool = False,
        params: dict[str, Any] | None = None,
    ):
        self.dataset_path = dataset_path
        self.model_type = model_type
        self.tag = tag or f"ml-{model_type}"
        self.max_samples = max_samples
        self.candles_dir = candles_dir or CANDLES_DIR
        self.serialize = serialize
        self.verbose = verbose
        self.params = params

    def _load_best_params(self) -> dict[str, Any] | None:
        name_map = {"lstm": "lstm", "xgboost": "xgboost", "random-forest": "random_forest"}
        fname = OPTIMIZATION_RESULTS_DIR / f"{name_map.get(self.model_type, self.model_type)}_optimization.json"
        if fname.exists():
            with open(fname) as f:
                data = json.load(f)
            params = data.get("best_params", {})
            print(f"  Loaded best params from {fname.name}: {params}")
            return params
        return None

    def run(self) -> dict[str, Any]:
        candles_map, ts_idx_map = _load_candles_map(self.candles_dir)

        params = self.params or self._load_best_params()
        predictor = get_predictor(self.model_type, params=params)
        print(f"Training {self.model_type} ...")
        start = time.time()
        train_info = predictor.train(self.dataset_path)
        print(
            f"  Train done in {time.time() - start:.1f}s  "
            f"val_acc={train_info['val_accuracy']:.3f}  "
            f"TFs={train_info.get('timeframes')}  features={train_info.get('n_features')}"
        )

        all_samples = _load_dataset(self.dataset_path)
        _train, _val, test = _temporal_split(all_samples)
        if self.max_samples:
            test = test[: self.max_samples]
        print(f"Testing on {len(test)} samples...")

        # Pre-build LSTM sequences over the full dataset for proper temporal context
        lstm_sequences = None
        if self.model_type == "lstm":
            import numpy as np
            import torch

            all_features = np.array(
                [extract_features(s["indicators"], predictor._timeframes) for s in all_samples], dtype=np.float32
            )
            all_features = (all_features - predictor._mean) / predictor._std
            seq_len = predictor.sequence_length
            n, d = all_features.shape
            seqs = np.zeros((n, seq_len, d), dtype=np.float32)
            for i in range(n):
                s = max(0, i - seq_len + 1)
                chunk = all_features[s : i + 1]
                seqs[i, seq_len - len(chunk) :] = chunk
            lstm_sequences = torch.tensor(seqs)

        test_offset = len(_train) + len(_val)
        predictions, actuals, trade_results, sample_keys = [], [], [], []
        errors = 0

        for i, sample in enumerate(test):
            symbol = sample.get("symbol", "BTCUSDT")
            label = sample["label"]
            atr = sample.get("atr_raw", 0)

            pred = (
                predictor.predict(sample["indicators"])
                if lstm_sequences is None
                else predictor.predict_sequence(lstm_sequences[test_offset + i].unsqueeze(0))
            )

            entry = label["entry"]
            if atr > 0:
                tp = entry + 1.5 * atr if pred["bias"] == "LONG" else entry - 1.5 * atr
                sl = entry - 1.0 * atr if pred["bias"] == "LONG" else entry + 1.0 * atr
            else:
                tp, sl = label["tp"], label["sl"]

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
            sample_keys.append(f"{symbol}@{sample['timestamp']}")

            candle_idx = ts_idx_map.get(symbol, {}).get(sample["timestamp"])
            if candle_idx is not None and candle_idx + 1 < len(candles_map.get(symbol, [])):
                future = candles_map[symbol][candle_idx + 1 : candle_idx + 25]
                trade_result = simulate_trade(prediction, future)
            else:
                trade_result = {"outcome": "ERROR", "pnl_pct": 0, "hold_bars": 0}
                errors += 1
            trade_results.append(trade_result)

            if self.verbose:
                ts = sample.get("timestamp", "")
                ts_str = f" @ {datetime.fromtimestamp(ts / 1000).strftime('%Y-%m-%d %H:%M')}" if ts else ""
                icon = {"WIN": "✅", "LOSS": "❌"}.get(trade_result["outcome"], "⏱️")
                match = "✓" if pred["bias"] == label["bias"] else "✗"
                print(
                    f"  [{i + 1}] {symbol}{ts_str} pred={pred['bias']} actual={label['bias']} {match} {icon} {trade_result['pnl_pct']:+.2f}%"
                )

        elapsed = time.time() - start
        metrics = compute_all_metrics(predictions, actuals, trade_results) if predictions else {}

        result = {
            "tag": self.tag,
            "provider": self.model_type,
            "model": self.model_type,
            "dataset": str(self.dataset_path),
            "total_samples": len(test),
            "successful_predictions": len(predictions),
            "errors": errors,
            "elapsed_seconds": round(elapsed, 2),
            "train_info": train_info,
            "metrics": metrics,
            "predictions": predictions,
            "actuals": actuals,
            "trade_results": trade_results,
            "sample_keys": sample_keys,
        }

        out = _save_result(result, self.tag)
        print(f"\nResults → {out}")

        if self.serialize:
            ext = "pt" if self.model_type == "lstm" else "pkl"
            model_path = OPTIMIZATION_RESULTS_DIR / f"{self.model_type.replace('-', '_')}_final.{ext}"
            predictor.save(str(model_path))
            print(f"Model → {model_path}")
            result["model_path"] = str(model_path)

        return result
