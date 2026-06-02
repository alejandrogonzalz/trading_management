"""Backtest framework public API."""

from backtest.config import DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES, DEFAULT_MONTHS


from backtest.models import (
    extract_features, get_predictor,
    XGBoostPredictor, RandomForestPredictor, LSTMPredictor,
)


def __getattr__(name):
    """Lazy-load heavy subpackages to avoid pulling langchain/ta-lib transitively."""
    _pipeline_names = {"DataPipeline"}
    _export_names = {"export_training_data", "build_training_example"}
    _ingestion_names = {
        "fetch_candles", "fetch_multi_tf_candles", "save_candles", "load_candles",
        "calculate_indicators_batch", "calculate_multi_tf_indicators",
        "label_candle", "generate_labeled_dataset", "save_labeled_dataset",
    }
    _eval_names = {
        "compute_all_metrics", "simulate_trade", "compare",
        "print_report", "print_comparison",
        "LLMBacktestRunner", "MLBacktestRunner",
    }
    if name in _pipeline_names:
        from backtest.pipeline import DataPipeline
        return DataPipeline
    if name in _export_names:
        from backtest import export
        return getattr(export, name)
    if name in _ingestion_names:
        from backtest import ingestion
        return getattr(ingestion, name)
    if name in _eval_names:
        from backtest import evaluation
        return getattr(evaluation, name)
    raise AttributeError(f"module 'backtest' has no attribute {name!r}")

__all__ = [
    "DEFAULT_SYMBOLS", "DEFAULT_TIMEFRAMES", "DEFAULT_MONTHS",
    "DataPipeline",
    "export_training_data", "build_training_example",
    "fetch_candles", "fetch_multi_tf_candles", "save_candles", "load_candles",
    "calculate_indicators_batch", "calculate_multi_tf_indicators",
    "label_candle", "generate_labeled_dataset", "save_labeled_dataset",
    "extract_features", "get_predictor",
    "XGBoostPredictor", "RandomForestPredictor", "LSTMPredictor",
    "compute_all_metrics", "simulate_trade", "compare",
    "print_report", "print_comparison",
    "LLMBacktestRunner", "MLBacktestRunner",
]
