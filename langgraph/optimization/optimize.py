#!/usr/bin/env python3
"""CLI entry point for hyperparameter optimization.

Usage:
    python optimize.py --model xgboost --config configs/xgboost.yaml
    python optimize.py --model lstm --config configs/lstm.yaml --dataset /path/to/data.jsonl
    nohup python optimize.py --model xgboost --config configs/xgboost.yaml > logs/xgb.log 2>&1 &
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimization.pipeline import DEFAULT_DATASET, RESULTS_DIR, OptimizerPipeline  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument(
        "--model",
        required=True,
        choices=["xgboost", "random_forest", "lstm", "qlora"],
    )
    parser.add_argument("--config", required=True, help="Path to YAML config")
    parser.add_argument(
        "--dataset",
        default=str(DEFAULT_DATASET),
        help="Path to labeled JSONL dataset",
    )
    parser.add_argument(
        "--output-dir",
        default=str(RESULTS_DIR),
        help="Results output directory",
    )
    args = parser.parse_args()

    pipeline = OptimizerPipeline(
        model=args.model,
        config_path=args.config,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
    )
    result = pipeline.run()

    print(f"\n{'=' * 50}")
    print(f"  {args.model.upper()} — Optimization Complete")
    print(f"{'=' * 50}")
    if result.get("best_score") is not None:
        print(f"  Best score : {result['best_score']:.4f}")
        print(f"  Best params: {json.dumps(result['best_params'], indent=4)}")
    if result.get("elapsed_seconds"):
        m, s = divmod(int(result["elapsed_seconds"]), 60)
        print(f"  Time       : {m}m {s}s")
    print(f"{'=' * 50}\n")


if __name__ == "__main__":
    main()
