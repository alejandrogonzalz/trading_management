"""Backtest framework public API."""

from backtest.config import DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES, DEFAULT_MONTHS
from backtest.pipeline import DataPipeline
from backtest.export import export_training_data, build_training_example

from backtest.ingestion import (
    fetch_candles, fetch_multi_tf_candles, save_candles, load_candles,
    calculate_indicators_batch, calculate_multi_tf_indicators,
    label_candle, generate_labeled_dataset, save_labeled_dataset,
)

from backtest.models import (
    extract_features, get_predictor,
    XGBoostPredictor, RandomForestPredictor, LSTMPredictor,
)

from backtest.evaluation import (
    compute_all_metrics, simulate_trade, compare,
    print_report, print_comparison,
    LLMBacktestRunner, MLBacktestRunner,
)

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
