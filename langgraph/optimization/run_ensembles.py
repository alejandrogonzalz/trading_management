#!/usr/bin/env python3
"""Train all ensemble models for Avance 5.

Uses the same class-based pipeline as the individual model optimization.
Runs sequentially, saves results to optimization/results/.

Usage:
    cd langgraph/
    OMP_NUM_THREADS=4 python optimization/run_ensembles.py
    # or in tmux:
    tmux new -s ensembles
    OMP_NUM_THREADS=4 python optimization/run_ensembles.py 2>&1 | tee logs/ensembles.log
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimization.searchers.ensemble_searcher import (
    AdaBoostSearcher,
    BaggingLSTMSearcher,
    BlendingSearcher,
    StackingSearcher,
    VotingSearcher,
)

DATASET = str(Path(__file__).resolve().parent.parent / "backtest" / "data" / "labeled" / "dataset.jsonl")
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Best params from Avance 4
ENSEMBLE_CONFIG = {
    "random_state": 42,
    "lstm_params": {"hidden_size": 32, "num_layers": 3, "sequence_length": 5},
    "xgb_params": {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.01,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 2.0,
        "min_child_weight": 3,
    },
    "n_bags": 5,
    "cv_splits": 5,
    "blend_fraction": 0.3,
    "param_grid": {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1, 0.5],
    },
}

SEARCHERS = [
    ("bagging_lstm", BaggingLSTMSearcher()),
    ("adaboost", AdaBoostSearcher()),
    ("soft_voting", VotingSearcher()),
    ("stacking", StackingSearcher()),
    ("blending", BlendingSearcher()),
]


def main():
    log.info("=" * 60)
    log.info("  ENSEMBLE OPTIMIZATION — Avance 5")
    log.info(f"  Started: {datetime.now().isoformat()}")
    log.info(f"  Dataset: {DATASET}")
    log.info("=" * 60)

    all_results = []
    t0_total = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    for name, searcher in SEARCHERS:
        out_path = RESULTS_DIR / f"{name}_optimization.json"

        if out_path.exists():
            log.info(f"\n  Skipping {name} — already completed ({out_path.name})")
            with open(out_path) as f:
                all_results.append(json.load(f))
            continue

        log.info(f"\n{'─' * 60}")
        log.info(f"  Running: {name}")
        log.info(f"{'─' * 60}")

        try:
            result = searcher.search(ENSEMBLE_CONFIG, DATASET)
            all_results.append(result)

            with open(out_path, "w") as f:
                json.dump(result, f, indent=2)
            log.info(f"  Saved: {out_path}")

        except Exception as e:
            log.error(f"  FAILED: {name} — {e}", exc_info=True)
            all_results.append({"model": name, "error": str(e)})

    # Summary
    elapsed_total = time.time() - t0_total
    log.info(f"\n{'=' * 60}")
    log.info("  FINAL SUMMARY")
    log.info(f"{'=' * 60}")
    log.info(f"  {'Model':<18} {'Type':<15} {'Val Acc':<10} {'Test Acc':<10} {'Time':<8}")
    log.info(f"  {'─' * 61}")

    for r in sorted(all_results, key=lambda x: x.get("test_metrics", {}).get("accuracy", 0), reverse=True):
        if "error" in r:
            log.info(f"  {r['model']:<18} FAILED: {r['error'][:40]}")
            continue
        log.info(
            f"  {r['model']:<18} {r.get('ensemble_type', '?'):<15} "
            f"{r['val_metrics']['accuracy']:<10.4f} "
            f"{r['test_metrics']['accuracy']:<10.4f} "
            f"{r['elapsed_seconds']:.0f}s"
        )

    log.info(f"\n  Total time: {elapsed_total:.1f}s ({elapsed_total / 60:.1f} min)")
    log.info(f"  Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
