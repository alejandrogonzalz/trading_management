"""Unit tests for the optimization package (searchers, io, pipeline)."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimization.io.results import load_all_results, save_result
from optimization.pipeline import OptimizerPipeline
from optimization.searchers.base import BaseSearcher
from optimization.searchers.qlora_searcher import QLoRASearcher

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_sample(bias: str = "LONG", ts: int = 1000) -> dict:
    ind = {
        tf: {
            "price": 50000.0,
            "rsi": 55.0,
            "macd_hist": 0.1,
            "adx": 20.0,
            "volume_ratio": 1.0,
            "atr_ratio": 0.01,
            "bb_pos": 0.5,
            "heatmap": "BULLISH",
            "structure": "BULLISH",
        }
        for tf in ["1h", "4h", "1d"]
    }
    return {
        "timestamp": ts,
        "symbol": "BTCUSDT",
        "indicators": ind,
        "label": {"bias": bias, "entry": 50000.0, "tp": 51000.0, "sl": 49000.0},
        "atr_raw": 500.0,
    }


@pytest.fixture
def tiny_dataset(tmp_path: Path) -> str:
    """Write a minimal JSONL dataset (40 samples, balanced) to a temp file."""
    samples = [_make_sample("LONG" if i % 2 == 0 else "SHORT", ts=i * 3600000) for i in range(40)]
    out = tmp_path / "dataset.jsonl"
    out.write_text("\n".join(json.dumps(s) for s in samples))
    return str(out)


@pytest.fixture
def xgboost_cfg() -> dict:
    return {
        "search_method": "random",
        "cv_splits": 2,
        "scoring": "accuracy",
        "n_random_iter": 2,
        "param_grid": {
            "n_estimators": [10],
            "max_depth": [2, 3],
        },
        "fixed_params": {"eval_metric": "logloss", "random_state": 42},
    }


@pytest.fixture
def rf_cfg() -> dict:
    return {
        "search_method": "random",
        "cv_splits": 2,
        "scoring": "accuracy",
        "n_random_iter": 2,
        "param_grid": {
            "n_estimators": [10],
            "max_depth": [2, 3],
        },
        "fixed_params": {"random_state": 42, "n_jobs": 1},
    }


@pytest.fixture
def lstm_cfg() -> dict:
    return {
        "search_method": "random",
        "n_random_iter": 1,
        "max_epochs": 2,
        "patience": 2,
        "param_grid": {
            "hidden_size": [8],
            "num_layers": [1],
            "sequence_length": [3],
            "learning_rate": [0.01],
            "dropout": [0.0],
            "batch_size": [8],
        },
    }


@pytest.fixture
def qlora_cfg() -> dict:
    return {
        "param_grid": {
            "learning_rate": [1e-4, 2e-4],
            "lora_rank": [16, 32],
        }
    }


# ---------------------------------------------------------------------------
# BaseSearcher
# ---------------------------------------------------------------------------


class TestBaseSearcher:
    def test_cannot_instantiate_abstract(self):
        with pytest.raises(TypeError):
            BaseSearcher()

    def test_concrete_subclass_must_implement_search(self):
        class Incomplete(BaseSearcher):
            pass

        with pytest.raises(TypeError):
            Incomplete()

    def test_concrete_subclass_ok(self):
        class Impl(BaseSearcher):
            def search(self, cfg, dataset_path):
                return {"model": "test"}

        s = Impl()
        assert s.search({}, "") == {"model": "test"}


# ---------------------------------------------------------------------------
# QLoRASearcher
# ---------------------------------------------------------------------------


class TestQLoRASearcher:
    def test_search_returns_standard_format(self, qlora_cfg, capsys):
        searcher = QLoRASearcher()
        result = searcher.search(qlora_cfg, "")

        assert result["model"] == "qlora"
        assert result["search_method"] == "manual"
        assert result["best_score"] is None
        assert result["best_params"] is None
        assert isinstance(result["all_results"], list)
        assert "timestamp" in result

    def test_search_picks_up_to_5_configs(self, qlora_cfg, capsys):
        searcher = QLoRASearcher()
        result = searcher.search(qlora_cfg, "")
        assert len(result["recommended_configs"]) <= 5

    def test_search_respects_recommended_configs(self, capsys):
        cfg = {
            "recommended_configs": [{"lr": 1e-4, "rank": 16}],
            "param_grid": {},
        }
        searcher = QLoRASearcher()
        result = searcher.search(cfg, "")
        assert result["recommended_configs"] == [{"lr": 1e-4, "rank": 16}]

    def test_search_prints_output(self, qlora_cfg, capsys):
        QLoRASearcher().search(qlora_cfg, "")
        captured = capsys.readouterr()
        assert "QLoRA" in captured.out


# ---------------------------------------------------------------------------
# SklearnSearcher
# ---------------------------------------------------------------------------


class TestSklearnSearcher:
    def test_xgboost_search_returns_standard_format(self, tiny_dataset, xgboost_cfg):
        from optimization.searchers.sklearn_searcher import SklearnSearcher

        searcher = SklearnSearcher("xgboost")
        result = searcher.search(xgboost_cfg, tiny_dataset)

        assert result["model"] == "xgboost"
        assert result["search_method"] == "random"
        assert 0 < result["best_score"] <= 1.0
        assert isinstance(result["best_params"], dict)
        assert "n_estimators" in result["best_params"] or "max_depth" in result["best_params"]
        assert len(result["all_results"]) > 0
        assert result["dataset_size"] > 0

    def test_random_forest_search_returns_standard_format(self, tiny_dataset, rf_cfg):
        from optimization.searchers.sklearn_searcher import SklearnSearcher

        searcher = SklearnSearcher("random_forest")
        result = searcher.search(rf_cfg, tiny_dataset)

        assert result["model"] == "random_forest"
        assert 0 < result["best_score"] <= 1.0
        assert isinstance(result["best_params"], dict)

    def test_invalid_model_raises(self):
        from optimization.searchers.sklearn_searcher import SklearnSearcher

        with pytest.raises(ValueError, match="Unknown sklearn model"):
            SklearnSearcher("lstm")

    def test_max_depth_minus_one_converted_to_none(self, tiny_dataset):
        from optimization.searchers.sklearn_searcher import SklearnSearcher

        cfg = {
            "search_method": "random",
            "cv_splits": 2,
            "n_random_iter": 1,
            "param_grid": {"n_estimators": [10], "max_depth": [-1]},
            "fixed_params": {"random_state": 42, "n_jobs": 1},
        }
        # Should not raise — -1 is converted to None before fitting
        searcher = SklearnSearcher("random_forest")
        result = searcher.search(cfg, tiny_dataset)
        assert result["model"] == "random_forest"

    def test_all_results_sorted_by_rank(self, tiny_dataset, xgboost_cfg):
        from optimization.searchers.sklearn_searcher import SklearnSearcher

        result = SklearnSearcher("xgboost").search(xgboost_cfg, tiny_dataset)
        ranks = [r["rank"] for r in result["all_results"]]
        assert ranks == sorted(ranks)


# ---------------------------------------------------------------------------
# LSTMSearcher
# ---------------------------------------------------------------------------


class TestLSTMSearcher:
    def test_search_returns_standard_format(self, tiny_dataset, lstm_cfg):
        pytest.importorskip("torch")
        from optimization.searchers.lstm_searcher import LSTMSearcher

        result = LSTMSearcher().search(lstm_cfg, tiny_dataset)

        assert result["model"] == "lstm"
        assert result["search_method"] == "random"
        assert 0 < result["best_score"] <= 1.0
        assert isinstance(result["best_params"], dict)
        assert "hidden_size" in result["best_params"]
        assert len(result["all_results"]) == 1  # n_random_iter=1

    def test_results_sorted_descending_by_score(self, tiny_dataset, lstm_cfg):
        pytest.importorskip("torch")
        from optimization.searchers.lstm_searcher import LSTMSearcher

        cfg = dict(lstm_cfg)
        cfg["param_grid"] = {
            "hidden_size": [8, 16],
            "num_layers": [1],
            "sequence_length": [3],
            "learning_rate": [0.01],
            "dropout": [0.0],
            "batch_size": [8],
        }
        cfg["n_random_iter"] = 2

        result = LSTMSearcher().search(cfg, tiny_dataset)
        scores = [r["mean_score"] for r in result["all_results"]]
        assert scores == sorted(scores, reverse=True)


# ---------------------------------------------------------------------------
# io.results
# ---------------------------------------------------------------------------


class TestIoResults:
    def _make_result(self, model: str = "xgboost") -> dict:
        return {
            "model": model,
            "best_params": {"n_estimators": 100},
            "best_score": 0.75,
            "all_results": [{"params": {"n_estimators": 100}, "mean_score": 0.75, "rank": 1}],
            "elapsed_seconds": 60.0,
            "dataset_size": 1000,
            "timestamp": "2026-01-01T00:00:00",
        }

    def test_save_result_creates_file(self, tmp_path):
        result = self._make_result()
        path = save_result(result, tmp_path)
        assert path.exists()
        assert path.name == "xgboost_optimization.json"

    def test_save_result_content_roundtrips(self, tmp_path):
        result = self._make_result()
        path = save_result(result, tmp_path)
        loaded = json.loads(path.read_text())
        assert loaded["model"] == "xgboost"
        assert loaded["best_score"] == 0.75

    def test_save_result_serializes_numpy(self, tmp_path):
        result = self._make_result()
        result["best_score"] = np.float64(0.75)
        result["best_params"]["n_estimators"] = np.int64(100)
        path = save_result(result, tmp_path)
        loaded = json.loads(path.read_text())
        assert isinstance(loaded["best_score"], float)
        assert isinstance(loaded["best_params"]["n_estimators"], int)

    def test_load_all_results_empty_dir(self, tmp_path):
        assert load_all_results(tmp_path) == {}

    def test_load_all_results_finds_files(self, tmp_path):
        for model in ("xgboost", "random_forest"):
            save_result(self._make_result(model), tmp_path)
        data = load_all_results(tmp_path)
        assert set(data.keys()) == {"xgboost", "random_forest"}

    def test_load_all_results_correct_content(self, tmp_path):
        save_result(self._make_result("lstm"), tmp_path)
        data = load_all_results(tmp_path)
        assert data["lstm"]["best_score"] == 0.75

    def test_save_result_creates_output_dir(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        result = self._make_result()
        path = save_result(result, nested)
        assert path.exists()


# ---------------------------------------------------------------------------
# io.plots
# ---------------------------------------------------------------------------


class TestIoPlots:
    def _make_result(self, model: str = "xgboost") -> dict:
        return {
            "model": model,
            "best_score": 0.76,
            "all_results": [{"params": {"n_estimators": 100 + i}, "mean_score": 0.75 + i * 0.001} for i in range(5)],
        }

    def test_save_optimization_plot_creates_file(self, tmp_path):
        from optimization.io.plots import save_optimization_plot

        result = self._make_result()
        out = tmp_path / "xgboost_optimization.png"
        save_optimization_plot(result, out)
        assert out.exists()

    def test_save_optimization_plot_skips_when_no_results(self, tmp_path, caplog):
        import logging

        from optimization.io.plots import save_optimization_plot

        result = {"model": "xgboost", "best_score": None, "all_results": []}
        out = tmp_path / "empty.png"
        with caplog.at_level(logging.INFO):
            save_optimization_plot(result, out)
        assert not out.exists()

    def test_save_comparison_plot_creates_file(self, tmp_path):
        from optimization.io.plots import save_comparison_plot

        data = {
            "xgboost": self._make_result("xgboost"),
            "random_forest": self._make_result("random_forest"),
        }
        out = tmp_path / "comparison.png"
        save_comparison_plot(data, out)
        assert out.exists()

    def test_save_comparison_plot_skips_when_no_scores(self, tmp_path, caplog):
        import logging

        from optimization.io.plots import save_comparison_plot

        data = {"qlora": {"model": "qlora", "best_score": None, "all_results": []}}
        out = tmp_path / "comparison.png"
        with caplog.at_level(logging.WARNING):
            save_comparison_plot(data, out)
        assert not out.exists()


# ---------------------------------------------------------------------------
# OptimizerPipeline
# ---------------------------------------------------------------------------


class TestOptimizerPipeline:
    def _make_yaml_cfg(self, tmp_path: Path) -> str:
        cfg = {
            "search_method": "random",
            "cv_splits": 2,
            "n_random_iter": 2,
            "param_grid": {"n_estimators": [10], "max_depth": [2]},
            "fixed_params": {"eval_metric": "logloss", "random_state": 42},
        }
        import yaml

        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.dump(cfg))
        return str(p)

    def test_get_searcher_xgboost(self, tmp_path):
        from optimization.searchers.sklearn_searcher import SklearnSearcher

        p = OptimizerPipeline("xgboost", self._make_yaml_cfg(tmp_path))
        assert isinstance(p._get_searcher(), SklearnSearcher)

    def test_get_searcher_random_forest(self, tmp_path):
        from optimization.searchers.sklearn_searcher import SklearnSearcher

        p = OptimizerPipeline("random_forest", self._make_yaml_cfg(tmp_path))
        assert isinstance(p._get_searcher(), SklearnSearcher)

    def test_get_searcher_lstm(self, tmp_path):
        pytest.importorskip("torch")
        from optimization.searchers.lstm_searcher import LSTMSearcher

        p = OptimizerPipeline("lstm", self._make_yaml_cfg(tmp_path))
        assert isinstance(p._get_searcher(), LSTMSearcher)

    def test_get_searcher_qlora(self, tmp_path):
        from optimization.searchers.qlora_searcher import QLoRASearcher

        p = OptimizerPipeline("qlora", self._make_yaml_cfg(tmp_path))
        assert isinstance(p._get_searcher(), QLoRASearcher)

    def test_get_searcher_unknown_raises(self, tmp_path):
        p = OptimizerPipeline("unknown_model", self._make_yaml_cfg(tmp_path))
        with pytest.raises(ValueError, match="Unknown model"):
            p._get_searcher()

    def test_run_calls_searcher_and_saves(self, tmp_path, tiny_dataset):
        cfg_path = self._make_yaml_cfg(tmp_path)
        fake_result = {
            "model": "xgboost",
            "best_params": {"n_estimators": 10},
            "best_score": 0.7,
            "all_results": [],
            "elapsed_seconds": 1.0,
            "dataset_size": 40,
            "timestamp": "2026-01-01T00:00:00",
        }
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = fake_result

        pipeline = OptimizerPipeline("xgboost", cfg_path, tiny_dataset, str(tmp_path))
        with patch.object(pipeline, "_get_searcher", return_value=mock_searcher):
            result = pipeline.run()

        mock_searcher.search.assert_called_once()
        assert result["model"] == "xgboost"
        assert (tmp_path / "xgboost_optimization.json").exists()

    def test_run_saves_json_with_correct_content(self, tmp_path, tiny_dataset):
        cfg_path = self._make_yaml_cfg(tmp_path)
        fake_result = {
            "model": "xgboost",
            "best_params": {"n_estimators": 10},
            "best_score": 0.72,
            "all_results": [{"params": {}, "mean_score": 0.72}],
            "elapsed_seconds": 2.0,
            "dataset_size": 40,
            "timestamp": "2026-01-01T00:00:00",
        }
        mock_searcher = MagicMock()
        mock_searcher.search.return_value = fake_result

        pipeline = OptimizerPipeline("xgboost", cfg_path, tiny_dataset, str(tmp_path))
        with patch.object(pipeline, "_get_searcher", return_value=mock_searcher):
            pipeline.run()

        saved = json.loads((tmp_path / "xgboost_optimization.json").read_text())
        assert saved["best_score"] == 0.72
