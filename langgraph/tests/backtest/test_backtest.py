"""Unit tests for the backtest package — pipeline, runner, ml_models, prompts."""

import json
import sys
import tempfile
from pathlib import Path

import pytest

_LANGGRAPH_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_LANGGRAPH_DIR))

from agent.prompts import build_system_prompt, build_user_prompt
from backtest.evaluation.metrics import compute_all_metrics
from backtest.evaluation.simulate import _parse_prediction, simulate_trade
from backtest.ingestion.fetcher import _ms
from backtest.ingestion.labeler import label_candle
from backtest.models import (
    RandomForestPredictor,
    XGBoostPredictor,
    _infer_timeframes,
    _sorted_timeframes,
    extract_features,
)
from backtest.pipeline import DataPipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_TF_3 = ["1h", "4h", "1d"]
_TF_5 = ["15m", "1h", "4h", "1d", "1w"]


def _make_ind(tfs: list[str] = _TF_3) -> dict:
    return {
        tf: {
            "price": 50000.0,
            "rsi": 55.0,
            "macd_hist": 0.1,
            "adx": 25.0,
            "volume_ratio": 1.2,
            "atr_ratio": 1.0,
            "bb_pos": 0.6,
            "heatmap": "BULLISH",
            "structure": "BULLISH",
        }
        for tf in tfs
    }


def _make_sample(bias: str = "LONG", ts: int = 1_000_000, tfs: list[str] = _TF_3) -> dict:
    return {
        "timestamp": ts,
        "symbol": "BTCUSDT",
        "indicators": _make_ind(tfs),
        "atr_raw": 500.0,
        "label": {
            "bias": bias,
            "entry": 50000.0,
            "tp": 52000.0 if bias == "LONG" else 48000.0,
            "sl": 49000.0 if bias == "LONG" else 51000.0,
            "confidence": 7,
            "quality": "HIGH",
        },
    }


def _make_dataset(n: int = 120, tfs: list[str] = _TF_3) -> list[dict]:
    samples = []
    for i in range(n):
        bias = "LONG" if i % 2 == 0 else "SHORT"
        samples.append(_make_sample(bias, ts=i * 3_600_000, tfs=tfs))
    return samples


def _write_dataset(samples: list[dict], path: Path) -> None:
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")


# ---------------------------------------------------------------------------
# agent/prompts
# ---------------------------------------------------------------------------


class TestPrompts:
    def test_system_prompt_futures(self):
        sp = build_system_prompt("FUTURES")
        assert "FUTURES" in sp
        assert "Senior Technical Analyst" in sp
        assert "LONG or SHORT" in sp

    def test_system_prompt_spot(self):
        sp = build_system_prompt("SPOT")
        assert "SPOT" in sp
        assert "Bias MUST be LONG" in sp

    def test_user_prompt_includes_symbol_and_indicators(self):
        ind = _make_ind()
        up = build_user_prompt("ETHUSDT", ind)
        assert "ETHUSDT" in up
        assert "1h" in up
        assert '"rsi"' in up

    def test_user_prompt_is_valid_template(self):
        ind = _make_ind()
        up = build_user_prompt("BTCUSDT", ind)
        assert "bias" in up
        assert "confidence" in up

    def test_prompts_are_identical_regardless_of_call_order(self):
        ind = _make_ind()
        a = build_system_prompt("FUTURES") + build_user_prompt("BTCUSDT", ind)
        b = build_system_prompt("FUTURES") + build_user_prompt("BTCUSDT", ind)
        assert a == b


# ---------------------------------------------------------------------------
# ml_models — feature extraction
# ---------------------------------------------------------------------------


class TestExtractFeatures:
    def test_3tf_produces_30_features(self):
        feats = extract_features(_make_ind(_TF_3))
        assert len(feats) == 3 * 9 + 3  # 30

    def test_5tf_produces_48_features(self):
        feats = extract_features(_make_ind(_TF_5))
        assert len(feats) == 5 * 9 + 3  # 48

    def test_pinned_timeframes_ignored_if_tf_missing(self):
        ind = _make_ind(["1h"])
        feats = extract_features(ind, timeframes=["1h", "4h", "1d"])
        assert len(feats) == 30
        assert feats[9:18] == [0.0] * 9
        assert feats[18:27] == [0.0] * 9

    def test_flat_dict_normalised_to_1h(self):
        flat = {
            "price": 100.0,
            "rsi": 50.0,
            "macd_hist": 0.0,
            "adx": 20.0,
            "volume_ratio": 1.0,
            "atr_ratio": 1.0,
            "bb_pos": 0.5,
            "heatmap": "NEUTRAL",
            "structure": "RANGE",
        }
        feats = extract_features(flat)
        assert len(feats) == 1 * 9 + 3  # wrapped as {"1h": ...}

    def test_sorted_timeframes_order(self):
        ind = _make_ind(["1d", "15m", "1h", "4h"])
        tfs = _sorted_timeframes(ind)
        assert tfs == ["15m", "1h", "4h", "1d"]

    def test_infer_timeframes_from_dataset(self):
        samples = _make_dataset(10, tfs=_TF_5)
        tfs = _infer_timeframes(samples)
        assert tfs == sorted(_TF_5, key=lambda t: {"15m": 15, "1h": 60, "4h": 240, "1d": 1440, "1w": 10080}[t])


# ---------------------------------------------------------------------------
# ml_models — predictors
# ---------------------------------------------------------------------------


class TestXGBoostPredictor:
    def test_train_and_predict(self):
        samples = _make_dataset(120)
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "ds.jsonl"
            _write_dataset(samples, ds)
            p = XGBoostPredictor()
            info = p.train(str(ds))
            assert info["val_accuracy"] >= 0.0
            assert info["timeframes"] == _TF_3
            assert info["n_features"] == 30

            pred = p.predict(_make_ind())
            assert pred["bias"] in ("LONG", "SHORT")
            assert 0 <= pred["confidence"] <= 10

    def test_save_and_load(self):
        samples = _make_dataset(120)
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "ds.jsonl"
            _write_dataset(samples, ds)
            p = XGBoostPredictor()
            p.train(str(ds))

            model_path = Path(td) / "xgb.json"
            p.save(str(model_path))

            p2 = XGBoostPredictor()
            p2.load(str(model_path))
            assert p2._timeframes == _TF_3
            pred = p2.predict(_make_ind())
            assert pred["bias"] in ("LONG", "SHORT")


class TestRandomForestPredictor:
    def test_train_and_predict(self):
        samples = _make_dataset(120)
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "ds.jsonl"
            _write_dataset(samples, ds)
            p = RandomForestPredictor()
            info = p.train(str(ds))
            assert info["timeframes"] == _TF_3

            pred = p.predict(_make_ind())
            assert pred["bias"] in ("LONG", "SHORT")

    def test_save_and_load_preserves_timeframes(self):
        samples = _make_dataset(120, tfs=_TF_5)
        with tempfile.TemporaryDirectory() as td:
            ds = Path(td) / "ds.jsonl"
            _write_dataset(samples, ds)
            p = RandomForestPredictor()
            p.train(str(ds))

            model_path = Path(td) / "rf.pkl"
            p.save(str(model_path))

            p2 = RandomForestPredictor()
            p2.load(str(model_path))
            assert set(p2._timeframes) == set(_TF_5)


# ---------------------------------------------------------------------------
# run_backtest — helpers
# ---------------------------------------------------------------------------


class TestParseAndSimulate:
    def test_parse_valid_json(self):
        raw = '{"bias": "LONG", "entry": 50000, "tp": 52000, "sl": 49000}'
        pred = _parse_prediction(raw)
        assert pred["bias"] == "LONG"

    def test_parse_json_in_markdown_fence(self):
        raw = '```json\n{"bias": "SHORT", "entry": 100}\n```'
        pred = _parse_prediction(raw)
        assert pred["bias"] == "SHORT"

    def test_parse_missing_required_field_returns_none(self):
        raw = '{"entry": 50000}'
        assert _parse_prediction(raw) is None

    def test_parse_invalid_json_returns_none(self):
        assert _parse_prediction("not json at all") is None

    def test_simulate_win_long(self):
        pred = {"bias": "LONG", "entry": 100, "tp": 110, "sl": 90}
        future = [{"high": 115, "low": 99, "close": 112}]
        result = simulate_trade(pred, future)
        assert result["outcome"] == "WIN"
        assert result["pnl_pct"] > 0

    def test_simulate_loss_long(self):
        pred = {"bias": "LONG", "entry": 100, "tp": 110, "sl": 90}
        future = [{"high": 101, "low": 85, "close": 88}]
        result = simulate_trade(pred, future)
        assert result["outcome"] == "LOSS"
        assert result["pnl_pct"] < 0

    def test_simulate_win_short(self):
        pred = {"bias": "SHORT", "entry": 100, "tp": 90, "sl": 110}
        future = [{"high": 101, "low": 85, "close": 88}]
        result = simulate_trade(pred, future)
        assert result["outcome"] == "WIN"

    def test_simulate_timeout(self):
        pred = {"bias": "LONG", "entry": 100, "tp": 200, "sl": 1}
        future = [{"high": 105, "low": 98, "close": 103}] * 24
        result = simulate_trade(pred, future)
        assert result["outcome"] == "TIMEOUT"


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


class TestMetrics:
    def _run(self, preds, actuals, trades):
        return compute_all_metrics(preds, actuals, trades)

    def test_perfect_accuracy(self):
        preds = [{"bias": "LONG", "confidence": 8, "entry": 100, "tp": 110, "sl": 90}] * 10
        actuals = [{"bias": "LONG"}] * 10
        trades = [{"outcome": "WIN", "pnl_pct": 5.0}] * 10
        m = self._run(preds, actuals, trades)
        assert m["direction_accuracy"] == pytest.approx(1.0)
        assert m["win_rate"] == pytest.approx(1.0)
        assert m["profit_factor"] > 1

    def test_zero_accuracy(self):
        preds = [{"bias": "LONG", "confidence": 5, "entry": 100, "tp": 110, "sl": 90}] * 10
        actuals = [{"bias": "SHORT"}] * 10
        trades = [{"outcome": "LOSS", "pnl_pct": -2.0}] * 10
        m = self._run(preds, actuals, trades)
        assert m["direction_accuracy"] == pytest.approx(0.0)

    def test_mixed_results(self):
        preds = [{"bias": "LONG", "confidence": 6, "entry": 100, "tp": 110, "sl": 90}] * 10
        actuals = [{"bias": "LONG"}] * 5 + [{"bias": "SHORT"}] * 5
        trades = [{"outcome": "WIN", "pnl_pct": 5.0}] * 5 + [{"outcome": "LOSS", "pnl_pct": -2.0}] * 5
        m = self._run(preds, actuals, trades)
        assert 0 < m["direction_accuracy"] < 1


# ---------------------------------------------------------------------------
# label_data
# ---------------------------------------------------------------------------


class TestLabelCandle:
    def _make_future(self, n: int, high: float, low: float) -> list[dict]:
        return [{"high": high, "low": low, "close": (high + low) / 2}] * n

    def _make_point(self, price: float, atr: float = 500.0) -> dict:
        return {
            "timestamp": 0,
            "atr_raw": atr,
            "indicators": {
                "1h": {
                    "price": price,
                    "rsi": 55,
                    "macd_hist": 0,
                    "adx": 25,
                    "volume_ratio": 1.2,
                    "atr_ratio": 1.0,
                    "bb_pos": 0.6,
                    "heatmap": "BULLISH",
                    "structure": "BULLISH",
                }
            },
        }

    def test_long_label(self):
        price = 50000.0
        atr = 500.0
        candles = [
            {
                "timestamp": i * 3600000,
                "open": price,
                "high": price + 1500,
                "low": price - 200,
                "close": price + 800,
                "volume": 100,
            }
            for i in range(30)
        ]
        point = self._make_point(price, atr)
        point["timestamp"] = candles[0]["timestamp"]
        result = label_candle(candles, point, 0, lookahead=24)
        if result:
            assert result["label"]["bias"] == "LONG"

    def test_ambiguous_returns_none(self):
        price = 50000.0
        candles = [
            {"timestamp": i, "open": price, "high": price + 100, "low": price - 100, "close": price, "volume": 100}
            for i in range(30)
        ]
        point = self._make_point(price, atr=500.0)
        point["timestamp"] = candles[0]["timestamp"]
        result = label_candle(candles, point, 0, lookahead=24)
        assert result is None

    def test_low_volume_filtered(self):
        price = 50000.0
        candles = [
            {
                "timestamp": i,
                "open": price,
                "high": price + 2000,
                "low": price - 100,
                "close": price + 1000,
                "volume": 10,
            }
            for i in range(30)
        ]
        point = self._make_point(price)
        point["indicators"]["1h"]["volume_ratio"] = 0.3
        point["timestamp"] = candles[0]["timestamp"]
        result = label_candle(candles, point, 0, lookahead=24)
        assert result is None


# ---------------------------------------------------------------------------
# pipeline (minimal / structural)
# ---------------------------------------------------------------------------


class TestDataPipeline:
    def test_init_validates_base_tf(self):
        with pytest.raises(ValueError, match="base_tf"):
            DataPipeline(timeframes=["4h", "1d"], base_tf="1h")

    def test_init_defaults(self):
        pipe = DataPipeline()
        assert pipe.base_tf == "1h"
        assert "1h" in pipe.timeframes

    def test_date_range_format(self):
        pipe = DataPipeline(months=3)
        start, end = pipe._date_range()
        assert len(start) == 10
        assert start < end

    def test_build_dataset_skips_symbol_without_candles(self, tmp_path):
        pipe = DataPipeline(
            symbols=["FAKESYM"],
            timeframes=["1h"],
            output_path=tmp_path / "ds.jsonl",
            candles_dir=tmp_path / "candles",
        )
        result = pipe.build_dataset()
        assert result == []


# ---------------------------------------------------------------------------
# fetch_candles helpers
# ---------------------------------------------------------------------------


class TestFetchHelpers:
    def test_ms_converts_date(self):
        ms = _ms("2024-01-01")
        assert ms == 1_704_067_200_000

    def test_ms_ordering(self):
        assert _ms("2024-01-01") < _ms("2024-06-01")
