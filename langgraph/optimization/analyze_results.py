#!/usr/bin/env python3
"""Compare optimization results across all models.

Usage:
    python analyze_results.py
    python analyze_results.py --results-dir /path/to/results
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from optimization.io.plots import save_comparison_plot  # noqa: E402
from optimization.io.results import load_all_results  # noqa: E402

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def print_summary(data: dict):
    print(f"\n{'=' * 70}")
    print("  Model Comparison Summary")
    print(f"{'=' * 70}")
    print(f"  {'Model':<18} {'Best Score':<12} {'Method':<10} {'Fits':<8} {'Time':<10}")
    print(f"  {'-' * 58}")
    for name, result in sorted(data.items(), key=lambda x: x[1].get("best_score") or 0, reverse=True):
        score = result.get("best_score")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        method = result.get("search_method", "?")
        fits = result.get("total_fits", 0)
        elapsed = result.get("elapsed_seconds", 0)
        m, s = divmod(int(elapsed), 60)
        time_str = f"{m}m {s}s" if elapsed else "N/A"
        print(f"  {name:<18} {score_str:<12} {method:<10} {fits:<8} {time_str:<10}")

    best = max(
        (v for v in data.values() if v.get("best_score")),
        key=lambda x: x["best_score"],
        default=None,
    )
    if best:
        print(f"\n  Best model: {best['model']} ({best['best_score']:.4f})")
        print(f"  Params: {json.dumps(best['best_params'], indent=6)}")
    print(f"{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze optimization results")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data = load_all_results(results_dir)

    if not data:
        log.error(f"No *_optimization.json files found in {results_dir}")
        return

    print_summary(data)
    save_comparison_plot(data, results_dir / "comparison.png")


if __name__ == "__main__":
    main()
