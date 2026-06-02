"""LSTM predictor for sequential indicator data."""

from pathlib import Path

import numpy as np

from backtest.models.features import _infer_timeframes, _load_dataset, _temporal_split, extract_features


class LSTMPredictor:
    """LSTM model that uses sequences of indicator vectors."""

    def __init__(self, sequence_length: int = 10, hidden_size: int = 64, num_layers: int = 2):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise ImportError("PyTorch is required for LSTM. Install with: pip install torch")
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None
        self.input_size: int | None = None
        self._timeframes: list[str] | None = None

    def _build_sequences(self, features: np.ndarray, labels: np.ndarray):
        import torch

        n, d = features.shape
        sequences = np.zeros((n, self.sequence_length, d), dtype=np.float32)
        for i in range(n):
            start = max(0, i - self.sequence_length + 1)
            seq = features[start : i + 1]
            sequences[i, self.sequence_length - len(seq) :] = seq
        return torch.tensor(sequences), torch.tensor(labels, dtype=torch.long)

    def train(self, dataset_path: str, val_split: float = 0.15) -> dict:
        import torch
        import torch.nn as nn

        samples = _load_dataset(dataset_path)
        self._timeframes = _infer_timeframes(samples)
        train_s, val_s, _test_s = _temporal_split(samples, 0.70, val_split)

        X_all = np.array([extract_features(s["indicators"], self._timeframes) for s in samples], dtype=np.float32)
        y_all = np.array([1 if s["label"]["bias"] == "LONG" else 0 for s in samples], dtype=np.int32)

        n_train = len(train_s)
        n_val = len(val_s)

        self._mean = X_all[:n_train].mean(axis=0)
        self._std = X_all[:n_train].std(axis=0) + 1e-8
        X_all = (X_all - self._mean) / self._std

        X_train_seq, y_train = self._build_sequences(X_all[:n_train], y_all[:n_train])
        X_val_seq, y_val = self._build_sequences(X_all[: n_train + n_val], y_all[: n_train + n_val])
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

        return {
            "train_size": n_train,
            "val_size": n_val,
            "val_accuracy": val_acc,
            "timeframes": self._timeframes,
            "n_features": self.input_size,
        }

    def predict(self, indicators: dict) -> dict:
        import torch

        X = np.array([extract_features(indicators, self._timeframes)], dtype=np.float32)
        X = (X - self._mean) / self._std
        seq = torch.tensor(X.reshape(1, 1, -1))
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
                "timeframes": self._timeframes,
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
        self._timeframes = ckpt.get("timeframes")
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
