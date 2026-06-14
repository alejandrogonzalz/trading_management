"""Feature extraction and dataset utilities shared by all ML models."""

import json

import numpy as np

_HEATMAP_MAP = {"STRONG_BEARISH": 0, "BEARISH": 1, "NEUTRAL": 2, "BULLISH": 3, "STRONG_BULLISH": 4}
_STRUCTURE_MAP = {"BEARISH": 0, "RANGE": 1, "BULLISH": 2, "BREAKOUT": 3}
_NUMERIC_KEYS = ["price", "rsi", "macd_hist", "adx", "volume_ratio", "atr_ratio", "bb_pos"]

_TF_MINUTES = {
    "1m": 1,
    "3m": 3,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
    "2h": 120,
    "4h": 240,
    "8h": 480,
    "1d": 1440,
    "3d": 4320,
    "1w": 10080,
    "1M": 43200,
}


def _sorted_timeframes(indicators: dict) -> list[str]:
    """Return TF keys present in indicators sorted from shortest to longest period."""
    tfs = [k for k, v in indicators.items() if isinstance(v, dict)]
    return sorted(tfs, key=lambda tf: _TF_MINUTES.get(tf, 9999))


def extract_features(indicators: dict, timeframes: list[str] | None = None) -> list[float]:
    """Convert multi-TF indicator dict to a flat feature vector.

    9 features per TF (7 numeric + 2 encoded categorical) + 3 cross-TF features.
    Missing timeframes are zero-filled so the vector length stays fixed.
    """
    if indicators and not isinstance(next(iter(indicators.values())), dict):
        indicators = {"1h": indicators}

    tfs = timeframes if timeframes is not None else _sorted_timeframes(indicators)

    features = []
    rsi_vals = []
    bullish_count = 0

    for tf in tfs:
        ind = indicators.get(tf, {})
        if not ind:
            features.extend([0.0] * 9)
            continue

        for key in _NUMERIC_KEYS:
            v = ind.get(key, 0.0)
            features.append(float(v) if v is not None else 0.0)

        features.append(float(_HEATMAP_MAP.get(ind.get("heatmap", "NEUTRAL"), 2)))
        features.append(float(_STRUCTURE_MAP.get(ind.get("structure", "RANGE"), 1)))

        rsi_vals.append(ind.get("rsi", 50.0))
        if ind.get("heatmap", "NEUTRAL") in ("BULLISH", "STRONG_BULLISH"):
            bullish_count += 1

    features.append(float(bullish_count))
    features.append(max(rsi_vals) - min(rsi_vals) if len(rsi_vals) >= 2 else 0.0)
    vr = [indicators.get(tf, {}).get("volume_ratio", 1.0) for tf in tfs if tf in indicators]
    features.append(max(vr) - min(vr) if len(vr) >= 2 else 0.0)

    return features


def _infer_timeframes(samples: list[dict]) -> list[str]:
    """Detect TF order from the first valid multi-TF sample."""
    for s in samples:
        ind = s.get("indicators", {})
        if ind and isinstance(next(iter(ind.values())), dict):
            return _sorted_timeframes(ind)
    return ["1h"]


def _load_dataset(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _temporal_split(
    samples: list[dict],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    embargo_bars: int = 24,
    base_tf_minutes: int = 60,
    sort: bool = True,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Strict temporal holdout: sort globally by timestamp and purge lookahead overlap.

    ``dataset.jsonl`` is grouped by symbol (``pipeline.build_dataset`` concatenates
    one symbol after another), so a positional cut splits *by symbol*, not by time.
    Sorting every sample by ``timestamp`` first gives a real holdout in time: train =
    oldest window, test = most recent window, across all symbols.

    The labeler looks ``embargo_bars`` candles into the future to assign each label.
    Any train sample whose labeling horizon crosses the start of val/test leaks future
    information, so we purge it. The purge is done by TIME (not by count) because, after
    the global sort, samples from all 12 symbols coexist near each boundary — dropping a
    fixed number of rows would not reliably cover the lookahead window.

    Timestamps are Binance klines in milliseconds (13 digits, ~1.7e12). ``embargo_bars``
    × ``base_tf_minutes`` defines the lookahead span; with ``base_tf="1h"`` that is
    ``base_tf_minutes=60``. ``sort=False`` preserves file order (only for callers that
    have already sorted).
    """
    if sort:
        samples = sorted(samples, key=lambda s: s.get("timestamp", 0))
    n = len(samples)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    embargo_ms = embargo_bars * base_tf_minutes * 60 * 1000  # timestamps are in ms

    val_start_ts = samples[train_end]["timestamp"] if train_end < n else None
    test_start_ts = samples[val_end]["timestamp"] if val_end < n else None

    train = samples[:train_end]
    val = samples[train_end:val_end]
    test = samples[val_end:]

    if val_start_ts is not None:
        train = [s for s in train if s.get("timestamp", 0) + embargo_ms <= val_start_ts]
    if test_start_ts is not None:
        val = [s for s in val if s.get("timestamp", 0) + embargo_ms <= test_start_ts]
    return train, val, test


def _samples_to_xy(samples: list[dict], timeframes: list[str] | None = None) -> tuple[np.ndarray, np.ndarray]:
    # Infer once so every sample is encoded with the same TF order and vector length.
    if timeframes is None:
        timeframes = _infer_timeframes(samples)
    X = np.array([extract_features(s["indicators"], timeframes) for s in samples], dtype=np.float32)
    y = np.array([1 if s["label"]["bias"] == "LONG" else 0 for s in samples], dtype=np.int32)
    return X, y
