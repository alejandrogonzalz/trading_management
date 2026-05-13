import logging
import time
from datetime import datetime

import numpy as np

from backtest.models.features import _load_dataset, _samples_to_xy, _temporal_split
from .base import BaseSearcher

log = logging.getLogger(__name__)


def _serialize(v):
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, np.floating):
        return float(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


class SklearnSearcher(BaseSearcher):
    """GridSearchCV / RandomizedSearchCV for XGBoost and Random Forest."""

    def __init__(self, model_name: str):
        if model_name not in ("xgboost", "random_forest"):
            raise ValueError(f"Unknown sklearn model: {model_name}")
        self.model_name = model_name

    def _build_estimator(self, cfg: dict):
        fixed = cfg.get("fixed_params", {})
        if self.model_name == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(
                eval_metric=fixed.get("eval_metric", "logloss"),
                random_state=fixed.get("random_state", 42),
            )
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            random_state=fixed.get("random_state", 42),
            n_jobs=fixed.get("n_jobs", -1),
        )

    def search(self, cfg: dict, dataset_path: str) -> dict:
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, TimeSeriesSplit

        samples = _load_dataset(dataset_path)
        train_s, val_s, _ = _temporal_split(samples, 0.70, 0.15)
        X, y = _samples_to_xy(train_s + val_s)
        log.info(f"Loaded {len(X)} samples for CV (train+val)")

        param_grid = dict(cfg["param_grid"])
        if "max_depth" in param_grid:
            param_grid["max_depth"] = [None if v == -1 else v for v in param_grid["max_depth"]]

        estimator = self._build_estimator(cfg)
        cv = TimeSeriesSplit(n_splits=cfg.get("cv_splits", 3))
        method = cfg.get("search_method", "grid")

        log.info(f"Starting {method} search for {self.model_name} | grid: {param_grid}")

        if method == "grid":
            search = GridSearchCV(
                estimator, param_grid, cv=cv,
                scoring=cfg.get("scoring", "accuracy"),
                verbose=1, n_jobs=-1, return_train_score=True,
            )
        else:
            search = RandomizedSearchCV(
                estimator, param_grid,
                n_iter=cfg.get("n_random_iter", 100), cv=cv,
                scoring=cfg.get("scoring", "accuracy"),
                verbose=1, n_jobs=-1, random_state=42, return_train_score=True,
            )

        t0 = time.time()
        search.fit(X, y)
        elapsed = time.time() - t0

        results = [
            {
                "params": search.cv_results_["params"][i],
                "mean_score": float(search.cv_results_["mean_test_score"][i]),
                "std_score": float(search.cv_results_["std_test_score"][i]),
                "mean_train_score": float(search.cv_results_["mean_train_score"][i]),
                "rank": int(search.cv_results_["rank_test_score"][i]),
            }
            for i in range(len(search.cv_results_["params"]))
        ]
        results.sort(key=lambda r: r["rank"])

        log.info(f"Best score: {search.best_score_:.4f} | Params: {search.best_params_}")
        log.info(f"Elapsed: {elapsed:.1f}s")

        return {
            "model": self.model_name,
            "search_method": method,
            "best_params": {k: _serialize(v) for k, v in search.best_params_.items()},
            "best_score": float(search.best_score_),
            "all_results": results[:50],
            "total_fits": len(results),
            "elapsed_seconds": round(elapsed, 1),
            "timestamp": datetime.now().isoformat(),
            "dataset_size": len(X),
            "cv_splits": cfg.get("cv_splits", 3),
        }
