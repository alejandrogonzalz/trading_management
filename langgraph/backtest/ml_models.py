"""Traditional ML models (XGBoost, Random Forest) for trade direction prediction."""

import json
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np

# Categorical encodings
_HEATMAP_MAP = {"STRONG_BEARISH": 0, "BEARISH": 1, "NEUTRAL": 2, "BULLISH": 3, "STRONG_BULLISH": 4}
_STRUCTURE_MAP = {"BEARISH": 0, "RANGE": 1, "BULLISH": 2, "BREAKOUT": 3}

MODELS_DIR = Path(__file__).parent / "data" / "models"

# Ordered numeric indicator keys per timeframe
_NUMERIC_KEYS = ["price", "rsi", "macd_hist", "adx", "volume_ratio", "atr_ratio", "bb_pos"]
_TIMEFRAMES = ["1h", "4h", "1d"]


def extract_features(indicators: dict) -> list[float]:
    """Convert multi-TF indicator dict to a flat feature vector.

    Input: {"1h": {"price": ..., "rsi": ..., ...}, "4h": {...}, "1d": {...}}
    Output: flat list of floats — 9 features per TF (7 numeric + 2 encoded categorical)
           + 3 derived cross-TF features.

    Missing timeframes are filled with zeros.
    """
    # Normalize: if flat dict (single-TF), wrap it
    if indicators and not isinstance(next(iter(indicators.values())), dict):
        indicators = {"1h": indicators}

    features = []
    rsi_vals = []
    bullish_count = 0

    for tf in _TIMEFRAMES:
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

    # Derived cross-TF features
    features.append(float(bullish_count))  # TF agreement
    features.append(max(rsi_vals) - min(rsi_vals) if len(rsi_vals) >= 2 else 0.0)  # RSI divergence
    # Volume ratio spread
    vr = [indicators.get(tf, {}).get("volume_ratio", 1.0) for tf in _TIMEFRAMES if tf in indicators]
    features.append(max(vr) - min(vr) if len(vr) >= 2 else 0.0)

    return features


def _load_dataset(path: str) -> list[dict]:
    samples = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def _temporal_split(
    samples: list[dict], train_frac: float = 0.70, val_frac: float = 0.15
) -> Tuple[list[dict], list[dict], list[dict]]:
    """Split samples temporally (already ordered by time in JSONL)."""
    n = len(samples)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return samples[:train_end], samples[train_end:val_end], samples[val_end:]


def _samples_to_xy(samples: list[dict]) -> Tuple[np.ndarray, np.ndarray]:
    X = np.array([extract_features(s["indicators"]) for s in samples], dtype=np.float32)
    y = np.array([1 if s["label"]["bias"] == "LONG" else 0 for s in samples], dtype=np.int32)
    return X, y


class XGBoostPredictor:
    def __init__(self):
        self.model = None
        self._feature_names = None

    def train(self, dataset_path: str, val_split: float = 0.15) -> dict:
        import xgboost as xgb

        samples = _load_dataset(dataset_path)
        train, val, _test = _temporal_split(samples, 0.70, val_split)

        X_train, y_train = _samples_to_xy(train)
        X_val, y_val = _samples_to_xy(val)

        self.model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            early_stopping_rounds=20,
            random_state=42,
        )
        self.model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        val_acc = float((self.model.predict(X_val) == y_val).mean())
        return {"train_size": len(train), "val_size": len(val), "val_accuracy": val_acc}

    def predict(self, indicators: dict) -> dict:
        X = np.array([extract_features(indicators)], dtype=np.float32)
        proba = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(proba))
        return {
            "bias": "LONG" if pred_class == 1 else "SHORT",
            "confidence": int(max(proba) * 10),
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(path)

    def load(self, path: str):
        import xgboost as xgb

        self.model = xgb.XGBClassifier()
        self.model.load_model(path)


class RandomForestPredictor:
    def __init__(self):
        self.model = None

    def train(self, dataset_path: str, val_split: float = 0.15) -> dict:
        from sklearn.ensemble import RandomForestClassifier

        samples = _load_dataset(dataset_path)
        train, val, _test = _temporal_split(samples, 0.70, val_split)

        X_train, y_train = _samples_to_xy(train)
        X_val, y_val = _samples_to_xy(val)

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        val_acc = float((self.model.predict(X_val) == y_val).mean())
        return {"train_size": len(train), "val_size": len(val), "val_accuracy": val_acc}

    def predict(self, indicators: dict) -> dict:
        X = np.array([extract_features(indicators)], dtype=np.float32)
        proba = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(proba))
        return {
            "bias": "LONG" if pred_class == 1 else "SHORT",
            "confidence": int(max(proba) * 10),
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            self.model = pickle.load(f)  # noqa: S301


def get_predictor(model_type: str):
    """Factory for model predictors."""
    models = {
        "xgboost": XGBoostPredictor,
        "random-forest": RandomForestPredictor,
    }
    if model_type not in models:
        raise ValueError(f"Unknown model: {model_type}. Choose from: {list(models.keys())}")
    return models[model_type]()
