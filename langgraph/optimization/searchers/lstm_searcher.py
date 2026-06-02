import itertools
import logging
import random
import time
from datetime import datetime

import numpy as np

from backtest.models.features import _load_dataset, _temporal_split, extract_features

from .base import BaseSearcher

log = logging.getLogger(__name__)


class LSTMSearcher(BaseSearcher):
    """Manual hyperparameter loop for LSTM with early stopping."""

    def search(self, cfg: dict, dataset_path: str) -> dict:
        import torch
        import torch.nn as nn
        from tqdm import tqdm

        samples = _load_dataset(dataset_path)
        train_s, val_s, _ = _temporal_split(samples, 0.70, 0.15)
        n_train, n_val = len(train_s), len(val_s)

        X_all = np.array([extract_features(s["indicators"]) for s in samples], dtype=np.float32)
        y_all = np.array([1 if s["label"]["bias"] == "LONG" else 0 for s in samples], dtype=np.int32)

        mean = X_all[:n_train].mean(axis=0)
        std = X_all[:n_train].std(axis=0) + 1e-8
        X_all = (X_all - mean) / std
        input_size = X_all.shape[1]

        device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
        )
        log.info(f"LSTM device: {device}")

        grid = cfg["param_grid"]
        all_combos = list(itertools.product(*grid.values()))
        keys = list(grid.keys())
        n_iter = cfg.get("n_random_iter", len(all_combos))
        if n_iter < len(all_combos):
            random.seed(42)
            all_combos = random.sample(all_combos, n_iter)

        log.info(f"LSTM: testing {len(all_combos)} configurations")
        max_epochs = cfg.get("max_epochs", 50)
        patience = cfg.get("patience", 10)
        results = []
        best_score = -1.0
        best_params = None
        t0 = time.time()

        def build_seq(features, labels, sl):
            n, d = features.shape
            seqs = np.zeros((n, sl, d), dtype=np.float32)
            for i in range(n):
                start = max(0, i - sl + 1)
                chunk = features[start : i + 1]
                seqs[i, sl - len(chunk) :] = chunk
            return (
                torch.tensor(seqs, device=device),
                torch.tensor(labels, dtype=torch.long, device=device),
            )

        for combo in tqdm(all_combos, desc="LSTM configs"):
            params = dict(zip(keys, combo))
            seq_len = params.get("sequence_length", 10)
            hidden = params.get("hidden_size", 64)
            layers = params.get("num_layers", 2)
            lr = params.get("learning_rate", 0.001)
            drop = params.get("dropout", 0.2)
            bs = params.get("batch_size", 32)

            X_tr, y_tr = build_seq(X_all[:n_train], y_all[:n_train], seq_len)
            X_va, y_va = build_seq(X_all[: n_train + n_val], y_all[: n_train + n_val], seq_len)
            X_va, y_va = X_va[n_train:], y_va[n_train:]

            model = self._build_net(input_size, hidden, layers, drop).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            best_val_loss = float("inf")
            wait = 0
            best_state = None
            epoch = 0

            for _epoch in range(max_epochs):
                model.train()
                perm = torch.randperm(len(X_tr), device=device)
                for i in range(0, len(perm), bs):
                    idx = perm[i : i + bs]
                    loss = criterion(model(X_tr[idx]), y_tr[idx])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                model.eval()
                with torch.no_grad():
                    vl = criterion(model(X_va), y_va).item()
                if vl < best_val_loss:
                    best_val_loss = vl
                    wait = 0
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                else:
                    wait += 1
                    if wait >= patience:
                        break

            if best_state:
                model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
            model.eval()
            with torch.no_grad():
                preds = model(X_va).argmax(dim=1).cpu().numpy()
            acc = float((preds == y_va.cpu().numpy()).mean())

            results.append({"params": params, "mean_score": acc, "epochs_trained": epoch + 1})
            if acc > best_score:
                best_score = acc
                best_params = params
                log.info(f"  New best: {acc:.4f} with {params}")

            del model, optimizer, X_tr, y_tr, X_va, y_va
            if device.type == "mps":
                torch.mps.empty_cache()
            elif device.type == "cuda":
                torch.cuda.empty_cache()

        elapsed = time.time() - t0
        results.sort(key=lambda r: r["mean_score"], reverse=True)
        log.info(f"Best LSTM score: {best_score:.4f} | Params: {best_params}")

        return {
            "model": "lstm",
            "search_method": "random",
            "best_params": best_params,
            "best_score": best_score,
            "all_results": results[:50],
            "total_fits": len(results),
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
            "dataset_size": n_train + n_val,
            "cv_splits": 1,
        }

    @staticmethod
    def _build_net(input_size, hidden, layers, drop):
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size,
                    hidden,
                    layers,
                    dropout=drop if layers > 1 else 0,
                    batch_first=True,
                )
                self.fc = nn.Linear(hidden, 2)

            def forward(self, x):
                out, _ = self.lstm(x)
                return self.fc(out[:, -1, :])

        return Net()
