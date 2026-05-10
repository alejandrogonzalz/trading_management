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


class LSTMPredictor:
    """LSTM model that uses sequences of indicator vectors."""

    def __init__(self, sequence_length=10, hidden_size=64, num_layers=2):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("PyTorch is required for LSTM. Install with: pip install torch")
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None
        self.input_size = None

    def _build_sequences(self, features: np.ndarray, labels: np.ndarray):
        """Build (seq_len, feature_dim) sequences with zero-padding for early samples."""
        import torch

        n, d = features.shape
        sequences = np.zeros((n, self.sequence_length, d), dtype=np.float32)
        for i in range(n):
            start = max(0, i - self.sequence_length + 1)
            seq = features[start : i + 1]
            # Right-align: pad zeros on the left for early samples
            sequences[i, self.sequence_length - len(seq) :] = seq
        return torch.tensor(sequences), torch.tensor(labels, dtype=torch.long)

    def train(self, dataset_path: str, val_split: float = 0.15) -> dict:
        import torch
        import torch.nn as nn

        samples = _load_dataset(dataset_path)
        train_s, val_s, _test_s = _temporal_split(samples, 0.70, val_split)

        X_all = np.array([extract_features(s["indicators"]) for s in samples], dtype=np.float32)
        y_all = np.array([1 if s["label"]["bias"] == "LONG" else 0 for s in samples], dtype=np.int32)

        n_train = len(train_s)
        n_val = len(val_s)

        # Normalize features using train stats
        self._mean = X_all[:n_train].mean(axis=0)
        self._std = X_all[:n_train].std(axis=0) + 1e-8
        X_all = (X_all - self._mean) / self._std

        X_train_seq, y_train = self._build_sequences(X_all[:n_train], y_all[:n_train])
        X_val_seq, y_val = self._build_sequences(X_all[: n_train + n_val], y_all[: n_train + n_val])
        # Only keep val portion
        X_val_seq, y_val = X_val_seq[n_train:], y_val[n_train:]

        self.input_size = X_all.shape[1]
        self.model = _LSTMNet(self.input_size, self.hidden_size, self.num_layers)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        batch_size = 32

        best_val_loss = float("inf")
        patience_counter = 0
        best_state = None

        self.model.train()
        for epoch in range(50):
            # Mini-batch training
            perm = torch.randperm(len(X_train_seq))
            epoch_loss = 0.0
            for i in range(0, len(perm), batch_size):
                idx = perm[i : i + batch_size]
                out = self.model(X_train_seq[idx])
                loss = criterion(out, y_train[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            epoch_loss /= len(X_train_seq)

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_out = self.model(X_val_seq)
                val_loss = criterion(val_out, y_val).item()
            self.model.train()

            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch + 1}/50  train_loss={epoch_loss:.4f}  val_loss={val_loss:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_state = {k: v.clone() for k, v in self.model.state_dict().items()}
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        if best_state:
            self.model.load_state_dict(best_state)

        self.model.eval()
        with torch.no_grad():
            val_preds = self.model(X_val_seq).argmax(dim=1).numpy()
        val_acc = float((val_preds == y_val.numpy()).mean())

        return {"train_size": n_train, "val_size": n_val, "val_accuracy": val_acc}

    def predict(self, indicators: dict) -> dict:
        """Predict from a single indicator dict (uses a 1-length sequence)."""
        import torch

        X = np.array([extract_features(indicators)], dtype=np.float32)
        X = (X - self._mean) / self._std
        seq = torch.tensor(X.reshape(1, 1, -1))  # (1, 1, features)
        # Zero-pad to sequence_length
        pad = torch.zeros(1, self.sequence_length - 1, seq.shape[2])
        seq = torch.cat([pad, seq], dim=1)

        self.model.eval()
        with torch.no_grad():
            proba = torch.softmax(self.model(seq), dim=1)[0].numpy()
        pred_class = int(np.argmax(proba))
        return {
            "bias": "LONG" if pred_class == 1 else "SHORT",
            "confidence": int(max(proba) * 10),
        }

    def predict_sequence(self, feature_sequence) -> dict:
        """Predict from a pre-built (1, seq_len, features) tensor."""
        import torch

        self.model.eval()
        with torch.no_grad():
            proba = torch.softmax(self.model(feature_sequence), dim=1)[0].numpy()
        pred_class = int(np.argmax(proba))
        return {
            "bias": "LONG" if pred_class == 1 else "SHORT",
            "confidence": int(max(proba) * 10),
        }

    def save(self, path: str):
        import torch

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.model.state_dict(),
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "sequence_length": self.sequence_length,
                "mean": self._mean,
                "std": self._std,
            },
            path,
        )

    def load(self, path: str):
        import torch

        ckpt = torch.load(path, weights_only=False)
        self.input_size = ckpt["input_size"]
        self.hidden_size = ckpt["hidden_size"]
        self.num_layers = ckpt["num_layers"]
        self.sequence_length = ckpt["sequence_length"]
        self._mean = ckpt["mean"]
        self._std = ckpt["std"]
        self.model = _LSTMNet(self.input_size, self.hidden_size, self.num_layers)
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()


class _LSTMNet:
    """Minimal LSTM wrapper using torch.nn."""

    def __new__(cls, input_size, hidden_size, num_layers):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=0.2 if num_layers > 1 else 0.0,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden_size, 2)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        return Net()


def get_predictor(model_type: str, params: dict = None):
    """Factory for model predictors. Pass params to override defaults (e.g. best params from optimization)."""
    if model_type not in ("xgboost", "random-forest", "lstm"):
        raise ValueError(f"Unknown model: {model_type}. Choose from: xgboost, random-forest, lstm")
    if model_type == "lstm":
        p = params or {}
        return LSTMPredictor(
            hidden_size=p.get("hidden_size", 64),
            num_layers=p.get("num_layers", 2),
            sequence_length=p.get("sequence_length", 10),
        )
    if model_type == "xgboost":
        return XGBoostPredictor()
    return RandomForestPredictor()
