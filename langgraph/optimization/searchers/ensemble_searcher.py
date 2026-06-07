"""Ensemble model searchers — Bagging, AdaBoost, Voting, Stacking, Blending."""

import gc
import logging
import time
from datetime import datetime

import numpy as np

from backtest.models.features import (
    _infer_timeframes,
    _load_dataset,
    _samples_to_xy,
    _temporal_split,
    extract_features,
)

from .base import BaseSearcher

log = logging.getLogger(__name__)


class _DataMixin:
    """Shared data loading logic for ensemble searchers."""

    def _load_all(self, dataset_path: str):
        samples = _load_dataset(dataset_path)
        timeframes = _infer_timeframes(samples)
        train_s, val_s, test_s = _temporal_split(samples, 0.70, 0.15)

        X_train, y_train = _samples_to_xy(train_s, timeframes)
        X_val, y_val = _samples_to_xy(val_s, timeframes)
        X_test, y_test = _samples_to_xy(test_s, timeframes)

        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(X_train)

        X_trainval = np.vstack([X_train, X_val])
        y_trainval = np.concatenate([y_train, y_val])

        return {
            "samples": samples,
            "timeframes": timeframes,
            "X_train": X_train,
            "y_train": y_train,
            "X_val": X_val,
            "y_val": y_val,
            "X_test": X_test,
            "y_test": y_test,
            "X_train_s": scaler.transform(X_train),
            "X_val_s": scaler.transform(X_val),
            "X_test_s": scaler.transform(X_test),
            "X_trainval": X_trainval,
            "y_trainval": y_trainval,
            "X_trainval_s": scaler.transform(X_trainval),
            "scaler": scaler,
            "n_train": len(train_s),
            "n_val": len(val_s),
        }

    def _evaluate(self, y_true, y_pred, y_proba=None):
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

        result = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        }
        if y_proba is not None:
            result["auc_roc"] = float(roc_auc_score(y_true, y_proba))
        return result

    def _build_lstm_sequences(self, X_norm, sl, start, end):
        import torch

        seqs = []
        for i in range(start, end):
            s = max(0, i - sl + 1)
            chunk = X_norm[s : i + 1]
            padded = np.zeros((sl, X_norm.shape[1]), dtype=np.float32)
            padded[sl - len(chunk) :] = chunk
            seqs.append(padded)
        return torch.tensor(np.array(seqs))


class BaggingLSTMSearcher(BaseSearcher, _DataMixin):
    """Train N LSTMs with different random seeds, combine via soft voting."""

    def search(self, cfg: dict, dataset_path: str) -> dict:
        import torch

        from backtest.models.lstm import LSTMPredictor

        data = self._load_all(dataset_path)
        lstm_params = cfg.get("lstm_params", {"hidden_size": 32, "num_layers": 3, "sequence_length": 5})
        n_bags = cfg.get("n_bags", 5)
        base_seed = cfg.get("random_state", 42)

        log.info(f"Bagging-LSTM: {n_bags} bags, params={lstm_params}")
        t0 = time.time()

        X_all = np.array(
            [extract_features(s["indicators"], data["timeframes"]) for s in data["samples"]],
            dtype=np.float32,
        )
        n_train, n_val = data["n_train"], data["n_val"]
        all_probas_val, all_probas_test = [], []

        for bag in range(n_bags):
            seed = base_seed + bag
            torch.manual_seed(seed)
            np.random.seed(seed)
            log.info(f"  Bag {bag + 1}/{n_bags} (seed={seed})")

            predictor = LSTMPredictor(**lstm_params)
            predictor.train(dataset_path)

            X_norm = (X_all - predictor._mean) / predictor._std
            sl = predictor.sequence_length

            predictor.model.eval()
            with torch.no_grad():
                val_proba = torch.softmax(
                    predictor.model(self._build_lstm_sequences(X_norm, sl, n_train, n_train + n_val)), dim=1
                )[:, 1].numpy()
                test_proba = torch.softmax(
                    predictor.model(self._build_lstm_sequences(X_norm, sl, n_train + n_val, len(data["samples"]))),
                    dim=1,
                )[:, 1].numpy()

            all_probas_val.append(val_proba)
            all_probas_test.append(test_proba)
            del predictor

        ensemble_val = np.mean(all_probas_val, axis=0)
        ensemble_test = np.mean(all_probas_test, axis=0)

        elapsed = time.time() - t0
        metrics_val = self._evaluate(data["y_val"], (ensemble_val > 0.5).astype(int), ensemble_val)
        metrics_test = self._evaluate(data["y_test"], (ensemble_test > 0.5).astype(int), ensemble_test)

        log.info(f"  Result: val={metrics_val['accuracy']:.4f} test={metrics_test['accuracy']:.4f} ({elapsed:.1f}s)")

        return {
            "model": "bagging_lstm",
            "ensemble_type": "homogeneous",
            "strategy": "bagging",
            "best_score": metrics_val["accuracy"],
            "best_params": {"n_bags": n_bags, **lstm_params},
            "val_metrics": metrics_val,
            "test_metrics": metrics_test,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }


class AdaBoostSearcher(BaseSearcher, _DataMixin):
    """AdaBoost with decision tree stumps + grid search."""

    def search(self, cfg: dict, dataset_path: str) -> dict:
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
        from sklearn.tree import DecisionTreeClassifier

        data = self._load_all(dataset_path)
        param_grid = cfg.get(
            "param_grid",
            {
                "n_estimators": [100, 200, 300],
                "learning_rate": [0.01, 0.05, 0.1, 0.5],
            },
        )
        random_state = cfg.get("random_state", 42)

        log.info(f"AdaBoost: grid={param_grid}")
        t0 = time.time()

        base = DecisionTreeClassifier(max_depth=2, random_state=random_state)
        import sklearn

        ada_kwargs = {"estimator": base, "random_state": random_state}
        if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) < (1, 4):
            ada_kwargs["algorithm"] = "SAMME"
        ada = AdaBoostClassifier(**ada_kwargs)

        tscv = TimeSeriesSplit(n_splits=cfg.get("cv_splits", 3))
        search = GridSearchCV(ada, param_grid, cv=tscv, scoring="accuracy", n_jobs=-1, verbose=0)
        search.fit(data["X_trainval_s"], data["y_trainval"])

        best = search.best_estimator_
        proba_val = best.predict_proba(data["X_val_s"])[:, 1]
        proba_test = best.predict_proba(data["X_test_s"])[:, 1]

        elapsed = time.time() - t0
        metrics_val = self._evaluate(data["y_val"], best.predict(data["X_val_s"]), proba_val)
        metrics_test = self._evaluate(data["y_test"], best.predict(data["X_test_s"]), proba_test)

        log.info(f"  Result: cv={search.best_score_:.4f} test={metrics_test['accuracy']:.4f} ({elapsed:.1f}s)")

        return {
            "model": "adaboost",
            "ensemble_type": "homogeneous",
            "strategy": "boosting",
            "best_score": float(search.best_score_),
            "best_params": search.best_params_,
            "val_metrics": metrics_val,
            "test_metrics": metrics_test,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }


class VotingSearcher(BaseSearcher, _DataMixin):
    """Soft voting: LSTM + XGBoost + SVM with accuracy-weighted probabilities."""

    def search(self, cfg: dict, dataset_path: str) -> dict:
        import torch
        import xgboost as xgb
        from sklearn.svm import SVC

        from backtest.models.lstm import LSTMPredictor

        data = self._load_all(dataset_path)
        lstm_params = cfg.get("lstm_params", {"hidden_size": 32, "num_layers": 3, "sequence_length": 5})
        xgb_params = cfg.get("xgb_params", {})
        random_state = cfg.get("random_state", 42)

        log.info("Soft Voting: LSTM + XGBoost + SVM")
        t0 = time.time()

        X_all = np.array(
            [extract_features(s["indicators"], data["timeframes"]) for s in data["samples"]],
            dtype=np.float32,
        )
        n_train, n_val = data["n_train"], data["n_val"]

        # LSTM
        torch.manual_seed(random_state)
        lstm = LSTMPredictor(**lstm_params)
        lstm.train(dataset_path)
        X_norm = (X_all - lstm._mean) / lstm._std
        sl = lstm.sequence_length

        lstm.model.eval()
        with torch.no_grad():
            lstm_val = torch.softmax(
                lstm.model(self._build_lstm_sequences(X_norm, sl, n_train, n_train + n_val)), dim=1
            )[:, 1].numpy()
            lstm_test = torch.softmax(
                lstm.model(self._build_lstm_sequences(X_norm, sl, n_train + n_val, len(data["samples"]))), dim=1
            )[:, 1].numpy()

        # XGBoost
        xgb_model = xgb.XGBClassifier(**xgb_params, eval_metric="logloss", random_state=random_state)
        xgb_model.fit(data["X_trainval"], data["y_trainval"])
        xgb_val = xgb_model.predict_proba(data["X_val"])[:, 1]
        xgb_test = xgb_model.predict_proba(data["X_test"])[:, 1]

        # SVM
        svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state)
        svm.fit(data["X_trainval_s"], data["y_trainval"])
        svm_val = svm.predict_proba(data["X_val_s"])[:, 1]
        svm_test = svm.predict_proba(data["X_test_s"])[:, 1]

        # Weights from individual val accuracy
        w = np.array(
            [
                float(((lstm_val > 0.5).astype(int) == data["y_val"]).mean()),
                float(((xgb_val > 0.5).astype(int) == data["y_val"]).mean()),
                float(((svm_val > 0.5).astype(int) == data["y_val"]).mean()),
            ]
        )
        w = w / w.sum()
        log.info(f"  Weights: LSTM={w[0]:.3f}, XGB={w[1]:.3f}, SVM={w[2]:.3f}")

        ensemble_val = w[0] * lstm_val + w[1] * xgb_val + w[2] * svm_val
        ensemble_test = w[0] * lstm_test + w[1] * xgb_test + w[2] * svm_test

        elapsed = time.time() - t0
        metrics_val = self._evaluate(data["y_val"], (ensemble_val > 0.5).astype(int), ensemble_val)
        metrics_test = self._evaluate(data["y_test"], (ensemble_test > 0.5).astype(int), ensemble_test)

        log.info(f"  Result: val={metrics_val['accuracy']:.4f} test={metrics_test['accuracy']:.4f} ({elapsed:.1f}s)")

        return {
            "model": "soft_voting",
            "ensemble_type": "heterogeneous",
            "strategy": "voting",
            "best_score": metrics_val["accuracy"],
            "best_params": {"weights": {"lstm": round(w[0], 4), "xgboost": round(w[1], 4), "svm": round(w[2], 4)}},
            "val_metrics": metrics_val,
            "test_metrics": metrics_test,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }


class StackingSearcher(BaseSearcher, _DataMixin):
    """2-level stacking with OOF predictions. Base: LSTM+XGB+SVM+MLP. Meta: LogReg."""

    def search(self, cfg: dict, dataset_path: str) -> dict:
        import torch
        import xgboost as xgb
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import TimeSeriesSplit
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC

        from backtest.models.lstm import LSTMPredictor

        data = self._load_all(dataset_path)
        lstm_params = cfg.get("lstm_params", {"hidden_size": 32, "num_layers": 3, "sequence_length": 5})
        xgb_params = cfg.get("xgb_params", {})
        random_state = cfg.get("random_state", 42)
        n_splits = cfg.get("cv_splits", 5)

        log.info(f"Stacking: {n_splits}-fold OOF, base=[LSTM, XGB, SVM, MLP], meta=LogReg")
        t0 = time.time()

        X_tv = data["X_trainval"]
        X_tv_s = data["X_trainval_s"]
        y_tv = data["y_trainval"]
        n_tv = len(y_tv)

        X_all = np.array(
            [extract_features(s["indicators"], data["timeframes"]) for s in data["samples"]],
            dtype=np.float32,
        )
        n_train, n_val = data["n_train"], data["n_val"]

        tscv = TimeSeriesSplit(n_splits=n_splits)
        oof = np.zeros((n_tv, 4))

        for fold, (tr_idx, vl_idx) in enumerate(tscv.split(X_tv)):
            log.info(f"  Fold {fold + 1}/{n_splits}")

            xgb_f = xgb.XGBClassifier(**xgb_params, eval_metric="logloss", random_state=random_state)
            xgb_f.fit(X_tv[tr_idx], y_tv[tr_idx])
            oof[vl_idx, 0] = xgb_f.predict_proba(X_tv[vl_idx])[:, 1]
            del xgb_f

            svm_f = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state)
            svm_f.fit(X_tv_s[tr_idx], y_tv[tr_idx])
            oof[vl_idx, 1] = svm_f.predict_proba(X_tv_s[vl_idx])[:, 1]
            del svm_f

            mlp_f = MLPClassifier(
                hidden_layer_sizes=(128, 64),
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.15,
                random_state=random_state,
            )
            mlp_f.fit(X_tv_s[tr_idx], y_tv[tr_idx])
            oof[vl_idx, 2] = mlp_f.predict_proba(X_tv_s[vl_idx])[:, 1]
            del mlp_f

            torch.manual_seed(random_state)
            lstm_f = LSTMPredictor(**lstm_params)
            lstm_f.train(dataset_path)
            X_norm = (X_all - lstm_f._mean) / lstm_f._std
            sl = lstm_f.sequence_length

            lstm_f.model.eval()
            with torch.no_grad():
                fold_seqs = self._build_lstm_sequences(X_norm, sl, vl_idx[0], vl_idx[-1] + 1)
                oof[vl_idx, 3] = torch.softmax(lstm_f.model(fold_seqs), dim=1)[:, 1].numpy()

            del lstm_f, X_norm, fold_seqs
            gc.collect()

        # Meta-learner
        valid_mask = oof.sum(axis=1) != 0
        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        meta.fit(oof[valid_mask], y_tv[valid_mask])

        # Final base learners on full trainval
        gc.collect()
        log.info("  Training final base learners...")
        torch.manual_seed(random_state)
        lstm_final = LSTMPredictor(**lstm_params)
        lstm_final.train(dataset_path)

        xgb_final = xgb.XGBClassifier(**xgb_params, eval_metric="logloss", random_state=random_state)
        xgb_final.fit(X_tv, y_tv)

        svm_final = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state)
        svm_final.fit(X_tv_s, y_tv)

        mlp_final = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=random_state,
        )
        mlp_final.fit(X_tv_s, y_tv)

        X_all_norm = (X_all - lstm_final._mean) / lstm_final._std
        sl = lstm_final.sequence_length
        lstm_final.model.eval()

        with torch.no_grad():
            lstm_val_p = torch.softmax(
                lstm_final.model(self._build_lstm_sequences(X_all_norm, sl, n_train, n_train + n_val)), dim=1
            )[:, 1].numpy()
            lstm_test_p = torch.softmax(
                lstm_final.model(self._build_lstm_sequences(X_all_norm, sl, n_train + n_val, len(data["samples"]))),
                dim=1,
            )[:, 1].numpy()

        val_meta = np.column_stack(
            [
                xgb_final.predict_proba(data["X_val"])[:, 1],
                svm_final.predict_proba(data["X_val_s"])[:, 1],
                mlp_final.predict_proba(data["X_val_s"])[:, 1],
                lstm_val_p,
            ]
        )
        test_meta = np.column_stack(
            [
                xgb_final.predict_proba(data["X_test"])[:, 1],
                svm_final.predict_proba(data["X_test_s"])[:, 1],
                mlp_final.predict_proba(data["X_test_s"])[:, 1],
                lstm_test_p,
            ]
        )

        proba_val = meta.predict_proba(val_meta)[:, 1]
        proba_test = meta.predict_proba(test_meta)[:, 1]

        elapsed = time.time() - t0
        metrics_val = self._evaluate(data["y_val"], meta.predict(val_meta), proba_val)
        metrics_test = self._evaluate(data["y_test"], meta.predict(test_meta), proba_test)

        log.info(f"  Result: val={metrics_val['accuracy']:.4f} test={metrics_test['accuracy']:.4f} ({elapsed:.1f}s)")
        log.info(f"  Meta coefs: {meta.coef_[0].tolist()}")

        return {
            "model": "stacking",
            "ensemble_type": "heterogeneous",
            "strategy": "stacking",
            "best_score": metrics_val["accuracy"],
            "best_params": {"meta_coefs": meta.coef_[0].tolist(), "base_learners": ["XGBoost", "SVM", "MLP", "LSTM"]},
            "val_metrics": metrics_val,
            "test_metrics": metrics_test,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }


class BlendingSearcher(BaseSearcher, _DataMixin):
    """Blending: train bases on 70% of trainval, meta on 30% blend set."""

    def search(self, cfg: dict, dataset_path: str) -> dict:
        import torch
        import xgboost as xgb
        from sklearn.linear_model import LogisticRegression
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC

        from backtest.models.lstm import LSTMPredictor

        data = self._load_all(dataset_path)
        lstm_params = cfg.get("lstm_params", {"hidden_size": 32, "num_layers": 3, "sequence_length": 5})
        xgb_params = cfg.get("xgb_params", {})
        random_state = cfg.get("random_state", 42)
        blend_frac = cfg.get("blend_fraction", 0.3)

        log.info(f"Blending: blend_fraction={blend_frac}")
        t0 = time.time()

        n_tv = len(data["y_trainval"])
        split_idx = int(n_tv * (1 - blend_frac))

        X_base, y_base = data["X_trainval"][:split_idx], data["y_trainval"][:split_idx]
        X_blend = data["X_trainval"][split_idx:]
        y_blend = data["y_trainval"][split_idx:]
        X_base_s = data["X_trainval_s"][:split_idx]
        X_blend_s = data["X_trainval_s"][split_idx:]

        X_all = np.array(
            [extract_features(s["indicators"], data["timeframes"]) for s in data["samples"]],
            dtype=np.float32,
        )
        n_train, n_val = data["n_train"], data["n_val"]

        # Base learners
        xgb_model = xgb.XGBClassifier(**xgb_params, eval_metric="logloss", random_state=random_state)
        xgb_model.fit(X_base, y_base)

        svm_model = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True, random_state=random_state)
        svm_model.fit(X_base_s, y_base)

        mlp_model = MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=300,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=random_state,
        )
        mlp_model.fit(X_base_s, y_base)

        torch.manual_seed(random_state)
        lstm_model = LSTMPredictor(**lstm_params)
        lstm_model.train(dataset_path)
        X_all_norm = (X_all - lstm_model._mean) / lstm_model._std
        sl = lstm_model.sequence_length
        lstm_model.model.eval()

        # Blend set meta-features
        with torch.no_grad():
            lstm_blend = torch.softmax(
                lstm_model.model(self._build_lstm_sequences(X_all_norm, sl, split_idx, n_tv)), dim=1
            )[:, 1].numpy()

        blend_meta = np.column_stack(
            [
                xgb_model.predict_proba(X_blend)[:, 1],
                svm_model.predict_proba(X_blend_s)[:, 1],
                mlp_model.predict_proba(X_blend_s)[:, 1],
                lstm_blend,
            ]
        )

        meta = LogisticRegression(C=1.0, max_iter=1000, random_state=random_state)
        meta.fit(blend_meta, y_blend)

        # Test & val predictions
        with torch.no_grad():
            lstm_val_p = torch.softmax(
                lstm_model.model(self._build_lstm_sequences(X_all_norm, sl, n_train, n_train + n_val)), dim=1
            )[:, 1].numpy()
            lstm_test_p = torch.softmax(
                lstm_model.model(self._build_lstm_sequences(X_all_norm, sl, n_train + n_val, len(data["samples"]))),
                dim=1,
            )[:, 1].numpy()

        val_meta = np.column_stack(
            [
                xgb_model.predict_proba(data["X_val"])[:, 1],
                svm_model.predict_proba(data["X_val_s"])[:, 1],
                mlp_model.predict_proba(data["X_val_s"])[:, 1],
                lstm_val_p,
            ]
        )
        test_meta = np.column_stack(
            [
                xgb_model.predict_proba(data["X_test"])[:, 1],
                svm_model.predict_proba(data["X_test_s"])[:, 1],
                mlp_model.predict_proba(data["X_test_s"])[:, 1],
                lstm_test_p,
            ]
        )

        proba_val = meta.predict_proba(val_meta)[:, 1]
        proba_test = meta.predict_proba(test_meta)[:, 1]

        elapsed = time.time() - t0
        metrics_val = self._evaluate(data["y_val"], meta.predict(val_meta), proba_val)
        metrics_test = self._evaluate(data["y_test"], meta.predict(test_meta), proba_test)

        log.info(f"  Result: val={metrics_val['accuracy']:.4f} test={metrics_test['accuracy']:.4f} ({elapsed:.1f}s)")

        return {
            "model": "blending",
            "ensemble_type": "heterogeneous",
            "strategy": "blending",
            "best_score": metrics_val["accuracy"],
            "best_params": {"blend_fraction": blend_frac, "base_learners": ["XGBoost", "SVM", "MLP", "LSTM"]},
            "val_metrics": metrics_val,
            "test_metrics": metrics_test,
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
        }
