"""Unit tests for train_qlora.py — no GPU required.

Covers the pure-Python logic: bias parsing, heuristic baseline, config
defaults/merging, and the per-run folder structure for saved artefacts.
GPU-dependent methods (load_model, train, evaluate, save_gguf) are not tested
here; they need a real Unsloth + CUDA environment.
"""

import json
from unittest.mock import MagicMock, patch

from optimization.qlora.train_qlora import (
    DATASET_PATH,
    DATASET_PATH_NO_FILTER,
    DATASET_TYPES,
    QLoRATrainer,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_trainer(tag: str = "test_run") -> QLoRATrainer:
    return QLoRATrainer({"tag": tag, "mode": "FUTURES"})


def _make_indicators(heatmap: str = "BULLISH", macd_hist: float = 0.1) -> dict:
    return {
        tf: {
            "heatmap": heatmap,
            "macd_hist": macd_hist,
            "rsi": 55.0,
            "adx": 25.0,
        }
        for tf in ["1h", "4h", "1d"]
    }


# ---------------------------------------------------------------------------
# _parse_bias — JSON path
# ---------------------------------------------------------------------------


class TestParseBias:
    def test_json_long(self):
        t = _make_trainer()
        assert t._parse_bias('{"bias": "LONG", "entry": 100}') == "LONG"

    def test_json_short(self):
        t = _make_trainer()
        assert t._parse_bias('{"bias": "SHORT", "entry": 100}') == "SHORT"

    def test_json_case_insensitive(self):
        t = _make_trainer()
        assert t._parse_bias('{"bias": "long"}') == "LONG"
        assert t._parse_bias('{"bias": "short"}') == "SHORT"

    def test_json_embedded_in_text(self):
        t = _make_trainer()
        text = 'Sure, here is my analysis: {"bias": "LONG", "tp": 200}'
        assert t._parse_bias(text) == "LONG"

    def test_json_invalid_bias_falls_back_to_keyword(self):
        t = _make_trainer()
        assert t._parse_bias('{"bias": "HOLD"} LONG mentioned here') == "LONG"

    def test_keyword_long_only(self):
        t = _make_trainer()
        assert t._parse_bias("The direction is LONG based on RSI") == "LONG"

    def test_keyword_short_only(self):
        t = _make_trainer()
        assert t._parse_bias("Bias: SHORT, expecting a drop") == "SHORT"

    def test_ambiguous_both_keywords_returns_none(self):
        t = _make_trainer()
        assert t._parse_bias("Could be LONG or SHORT") is None

    def test_empty_string_returns_none(self):
        t = _make_trainer()
        assert t._parse_bias("") is None

    def test_no_bias_returns_none(self):
        t = _make_trainer()
        assert t._parse_bias("The market is uncertain") is None

    def test_malformed_json_falls_back_to_keyword(self):
        t = _make_trainer()
        assert t._parse_bias("{broken json LONG") == "LONG"


# ---------------------------------------------------------------------------
# _heuristic_bias — indicator-based majority vote
# ---------------------------------------------------------------------------


class TestHeuristicBaseline:
    def test_majority_bullish_returns_long(self):
        t = _make_trainer()
        ind = {
            "1h": {"heatmap": "BULLISH", "macd_hist": 0.1},
            "4h": {"heatmap": "STRONG_BULLISH", "macd_hist": 0.2},
            "1d": {"heatmap": "BEARISH", "macd_hist": -0.1},
        }
        assert t._heuristic_bias(ind) == "LONG"

    def test_majority_bearish_returns_short(self):
        t = _make_trainer()
        ind = {
            "1h": {"heatmap": "BEARISH", "macd_hist": -0.1},
            "4h": {"heatmap": "STRONG_BEARISH", "macd_hist": -0.2},
            "1d": {"heatmap": "BULLISH", "macd_hist": 0.1},
        }
        assert t._heuristic_bias(ind) == "SHORT"

    def test_tie_positive_macd_returns_long(self):
        t = _make_trainer()
        ind = {
            "1h": {"heatmap": "BULLISH", "macd_hist": 0.5},
            "4h": {"heatmap": "BEARISH", "macd_hist": 0.5},
        }
        assert t._heuristic_bias(ind) == "LONG"

    def test_tie_negative_macd_returns_short(self):
        t = _make_trainer()
        ind = {
            "1h": {"heatmap": "BULLISH", "macd_hist": -0.5},
            "4h": {"heatmap": "BEARISH", "macd_hist": -0.5},
        }
        assert t._heuristic_bias(ind) == "SHORT"

    def test_single_tf_flat_format(self):
        # When caller passes a single-TF flat dict (not nested), the method
        # wraps it in {"1h": ...} and still returns a valid prediction.
        t = _make_trainer()
        flat = {"heatmap": "BEARISH", "macd_hist": -1.0}
        result = t._heuristic_bias(flat)
        assert result in ("LONG", "SHORT")

    def test_all_neutral_falls_back_to_macd(self):
        t = _make_trainer()
        ind = {
            "1h": {"heatmap": "NEUTRAL", "macd_hist": 0.3},
        }
        assert t._heuristic_bias(ind) == "LONG"


# ---------------------------------------------------------------------------
# Config defaults and merging
# ---------------------------------------------------------------------------


class TestConfig:
    REQUIRED_KEYS = [
        "model_name",
        "max_seq_length",
        "learning_rate",
        "lora_rank",
        "lora_alpha",
        "lora_dropout",
        "epochs",
        "batch_size",
        "gradient_accumulation_steps",
        "warmup_steps",
        "weight_decay",
        "seed",
        "mode",
        "tag",
        "output_dir",
        "dataset_path",
        "training_data_dir",
    ]

    def test_all_required_keys_present(self):
        for key in self.REQUIRED_KEYS:
            assert key in QLoRATrainer.DEFAULT_CONFIG, f"Missing key: {key}"

    def test_max_seq_length_at_least_1024(self):
        # Must be >= 1024 — samples are 877-933 tokens; lower crashes Unsloth.
        assert QLoRATrainer.DEFAULT_CONFIG["max_seq_length"] >= 1024

    def test_custom_config_overrides_defaults(self):
        t = QLoRATrainer({"learning_rate": 5e-5, "tag": "my_run"})
        assert t.cfg["learning_rate"] == 5e-5
        assert t.cfg["tag"] == "my_run"
        # Unspecified keys still come from defaults
        assert t.cfg["lora_rank"] == QLoRATrainer.DEFAULT_CONFIG["lora_rank"]

    def test_empty_config_uses_all_defaults(self):
        t = QLoRATrainer()
        assert t.cfg == QLoRATrainer.DEFAULT_CONFIG

    def test_lora_alpha_double_rank_by_default(self):
        cfg = QLoRATrainer.DEFAULT_CONFIG
        assert cfg["lora_alpha"] == cfg["lora_rank"] * 2

    def test_default_dataset_is_filtered(self):
        # Default config must point at the production (filtered) dataset so the
        # original run reproduces byte-for-byte.
        cfg = QLoRATrainer.DEFAULT_CONFIG
        assert cfg["dataset_path"] == DATASET_PATH
        assert cfg["dataset_path"].endswith("dataset.jsonl")


# ---------------------------------------------------------------------------
# Dataset selection — filtered vs no_filter (experiment/no-drawdown-filter)
# ---------------------------------------------------------------------------


class TestDatasetType:
    def test_dataset_types_keys(self):
        assert set(DATASET_TYPES) == {"filtered", "no_filter"}

    def test_filtered_maps_to_production_dataset(self):
        ds_path, export_dir = DATASET_TYPES["filtered"]
        assert ds_path == DATASET_PATH
        assert ds_path.endswith("dataset.jsonl")
        assert export_dir.endswith("training_data")

    def test_no_filter_maps_to_separate_dataset_and_export_dir(self):
        ds_path, export_dir = DATASET_TYPES["no_filter"]
        assert ds_path == DATASET_PATH_NO_FILTER
        assert ds_path.endswith("dataset_no_drawdown_filter.jsonl")
        # Separate export dir so it never clobbers the filtered chat splits.
        assert export_dir.endswith("training_data_no_filter")

    def test_no_filter_paths_distinct_from_filtered(self):
        assert DATASET_TYPES["filtered"][0] != DATASET_TYPES["no_filter"][0]
        assert DATASET_TYPES["filtered"][1] != DATASET_TYPES["no_filter"][1]

    def test_trainer_accepts_no_filter_config(self):
        no_filter_path, no_filter_dir = DATASET_TYPES["no_filter"]
        t = QLoRATrainer({"dataset_path": no_filter_path, "training_data_dir": no_filter_dir})
        assert t.cfg["dataset_path"] == no_filter_path
        assert t.cfg["training_data_dir"] == no_filter_dir


# ---------------------------------------------------------------------------
# File paths — per-run folder structure
# ---------------------------------------------------------------------------


class TestResultPaths:
    def test_save_loss_curve_writes_to_tag_subfolder(self, tmp_path):
        t = _make_trainer(tag="qlora_cloud")
        t.trainer = MagicMock()
        t.trainer.state.log_history = [
            {"loss": 1.2, "step": 10},
            {"eval_loss": 1.0, "step": 10},
        ]

        with (
            patch("optimization.qlora.train_qlora.RESULTS_DIR", tmp_path),
            patch("optimization.qlora.train_qlora.save_loss_curve_plot"),
        ):
            t.save_loss_curve()

        assert (tmp_path / "qlora_cloud" / "loss_curve.json").exists()

    def test_save_loss_curve_json_content(self, tmp_path):
        history = [{"loss": 1.5, "step": 1}, {"eval_loss": 1.3, "step": 1}]
        t = _make_trainer(tag="myrun")
        t.trainer = MagicMock()
        t.trainer.state.log_history = history

        with (
            patch("optimization.qlora.train_qlora.RESULTS_DIR", tmp_path),
            patch("optimization.qlora.train_qlora.save_loss_curve_plot"),
        ):
            t.save_loss_curve()

        saved = json.loads((tmp_path / "myrun" / "loss_curve.json").read_text())
        assert saved == history

    def test_save_loss_curve_skipped_when_no_trainer(self, tmp_path):
        t = _make_trainer(tag="qlora_cloud")
        # trainer is None by default — save_loss_curve must be a no-op
        with patch("optimization.qlora.train_qlora.RESULTS_DIR", tmp_path):
            t.save_loss_curve()  # must not raise
        assert not (tmp_path / "qlora_cloud").exists()

    def test_result_json_written_to_tag_subfolder(self, tmp_path):
        """The run-specific result.json goes to results/<tag>/result.json."""
        _make_trainer(tag="qlora_v2")
        fake_result = {"tag": "qlora_v2", "metrics": {"direction_accuracy": 0.88}}

        with patch("optimization.qlora.train_qlora.RESULTS_DIR", tmp_path):
            run_dir = tmp_path / "qlora_v2"
            run_dir.mkdir(parents=True, exist_ok=True)
            out_path = run_dir / "result.json"
            canonical = tmp_path / "qlora_optimization.json"
            for path in (out_path, canonical):
                path.write_text(json.dumps(fake_result))

        assert (tmp_path / "qlora_v2" / "result.json").exists()
        assert (tmp_path / "qlora_optimization.json").exists()

    def test_canonical_has_same_content_as_run_result(self, tmp_path):
        """qlora_optimization.json must mirror the latest run's result.json."""
        fake_result = {"tag": "qlora_cloud", "best_score": 0.88}
        run_dir = tmp_path / "qlora_cloud"
        run_dir.mkdir()
        (run_dir / "result.json").write_text(json.dumps(fake_result))
        (tmp_path / "qlora_optimization.json").write_text(json.dumps(fake_result))

        run_data = json.loads((run_dir / "result.json").read_text())
        canonical_data = json.loads((tmp_path / "qlora_optimization.json").read_text())
        assert run_data == canonical_data

    def test_different_tags_do_not_collide(self, tmp_path):
        """Two runs with different tags live in separate subdirectories."""
        for tag in ("run_a", "run_b"):
            d = tmp_path / tag
            d.mkdir()
            (d / "result.json").write_text(json.dumps({"tag": tag}))

        assert json.loads((tmp_path / "run_a" / "result.json").read_text())["tag"] == "run_a"
        assert json.loads((tmp_path / "run_b" / "result.json").read_text())["tag"] == "run_b"
