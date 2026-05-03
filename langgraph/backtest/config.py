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

DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
DEFAULT_MONTHS = 6
