"""XGBoost and Random Forest predictors."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np

from backtest.models.features import (
    _infer_timeframes,
    _load_dataset,
    _samples_to_xy,
    _temporal_split,
    extract_features,
)

MODELS_DIR = Path(__file__).parent.parent / "data" / "models"


class XGBoostPredictor:
    def __init__(self):
        self.model = None
        self._timeframes: list[str] | None = None

    def train(self, dataset_path: str, val_split: float = 0.15) -> dict:
        import xgboost as xgb

        samples = _load_dataset(dataset_path)
        self._timeframes = _infer_timeframes(samples)
        train, val, _test = _temporal_split(samples, 0.70, val_split)

        X_train, y_train = _samples_to_xy(train, self._timeframes)
        X_val, y_val = _samples_to_xy(val, self._timeframes)

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
        return {
            "train_size": len(train),
            "val_size": len(val),
            "val_accuracy": val_acc,
            "timeframes": self._timeframes,
            "n_features": X_train.shape[1],
        }

    def predict(self, indicators: dict) -> dict:
        X = np.array([extract_features(indicators, self._timeframes)], dtype=np.float32)
        proba = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(proba))
        return {
            "bias": "LONG" if pred_class == 1 else "SHORT",
            "confidence": int(max(proba) * 10),
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        meta_path = Path(path).with_suffix(".meta.pkl")
        self.model.save_model(path)
        with open(meta_path, "wb") as f:
            pickle.dump({"timeframes": self._timeframes}, f)

    def load(self, path: str):
        import xgboost as xgb

        self.model = xgb.XGBClassifier()
        self.model.load_model(path)
        meta_path = Path(path).with_suffix(".meta.pkl")
        if meta_path.exists():
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)  # noqa: S301
            self._timeframes = meta.get("timeframes")


class RandomForestPredictor:
    def __init__(self):
        self.model = None
        self._timeframes: list[str] | None = None

    def train(self, dataset_path: str, val_split: float = 0.15) -> dict:
        from sklearn.ensemble import RandomForestClassifier

        samples = _load_dataset(dataset_path)
        self._timeframes = _infer_timeframes(samples)
        train, val, _test = _temporal_split(samples, 0.70, val_split)

        X_train, y_train = _samples_to_xy(train, self._timeframes)
        X_val, y_val = _samples_to_xy(val, self._timeframes)

        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_train, y_train)

        val_acc = float((self.model.predict(X_val) == y_val).mean())
        return {
            "train_size": len(train),
            "val_size": len(val),
            "val_accuracy": val_acc,
            "timeframes": self._timeframes,
            "n_features": X_train.shape[1],
        }

    def predict(self, indicators: dict) -> dict:
        X = np.array([extract_features(indicators, self._timeframes)], dtype=np.float32)
        proba = self.model.predict_proba(X)[0]
        pred_class = int(np.argmax(proba))
        return {
            "bias": "LONG" if pred_class == 1 else "SHORT",
            "confidence": int(max(proba) * 10),
        }

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "timeframes": self._timeframes}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301
        if isinstance(data, dict):
            self.model = data["model"]
            self._timeframes = data.get("timeframes")
        else:
            self.model = data


def get_predictor(model_type: str, params: dict[str, Any] | None = None):
    """Factory for model predictors."""
    if model_type == "lstm":
        from backtest.models.lstm import LSTMPredictor

        p = params or {}
        return LSTMPredictor(
            hidden_size=p.get("hidden_size", 64),
            num_layers=p.get("num_layers", 2),
            sequence_length=p.get("sequence_length", 10),
        )
    if model_type == "xgboost":
        return XGBoostPredictor()
    if model_type == "random-forest":
        return RandomForestPredictor()
    raise ValueError(f"Unknown model: {model_type}. Choose from: xgboost, random-forest, lstm")
