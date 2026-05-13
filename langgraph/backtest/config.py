"""Shared configuration for backtest and training data generation."""

# Default symbols for backtesting and training data.
# These cover large, mid, and small cap for model generalization.
# Production scanner discovers pairs dynamically — this list is only for offline backtest.
DEFAULT_SYMBOLS = [
    # Large cap (stable, clear trends)
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    # Mid cap (more volatile)
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "AVAXUSDT",
    "DOTUSDT",
    # High volatility (diverse patterns)
    "DOGEUSDT",
    "LINKUSDT",
    "MATICUSDT",
    "NEARUSDT",
]

# All timeframes available in production (market_service.py).
# 5m/15m produce large files (~130K candles each over 18 months).
# 1h is the base TF for labeling — always required.
ALL_TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d", "1w"]

# Default for CLI commands — matches production minus very short TFs for manageability.
# Override with --timeframes to use ALL_TIMEFRAMES.
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h", "1d", "1w"]

DEFAULT_MONTHS = 18
