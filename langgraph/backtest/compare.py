"""Side-by-side comparison of two backtest runs."""

import json
from pathlib import Path
from typing import Dict, Any


def load_results(path: str) -> Dict[str, Any]:
    """Load a backtest results JSON file."""
    with open(path) as f:
        return json.load(f)


def compare(baseline_path: str, candidate_path: str) -> Dict[str, Any]:
    """Compare two backtest result files and return a comparison dict."""
    baseline = load_results(baseline_path)
    candidate = load_results(candidate_path)

    bm = baseline.get("metrics", {})
    cm = candidate.get("metrics", {})

    def _delta(b, c):
        if b is None or c is None:
            return None
        return round(c - b, 4)

    rows = []
    metric_keys = [
        ("direction_accuracy", "Direction Accuracy", True),
        ("win_rate", "Win Rate", True),
        ("profit_factor", "Profit Factor", True),
        ("avg_win", "Avg Win %", True),
        ("avg_loss", "Avg Loss %", False),  # Less negative is better
        ("sharpe_ratio", "Sharpe Ratio", True),
        ("max_drawdown", "Max Drawdown %", False),  # Lower is better
    ]

    for key, label, higher_is_better in metric_keys:
        bv = bm.get(key, 0)
        cv = cm.get(key, 0)
        delta = _delta(bv, cv)
        improved = (delta > 0) == higher_is_better if delta is not None else None
        rows.append({
            "metric": label,
            "baseline": bv,
            "candidate": cv,
            "delta": delta,
            "improved": improved,
        })

    # Per-class metrics
    for cls in ("long", "short"):
        bclass = bm.get(f"{cls}_metrics", {})
        cclass = cm.get(f"{cls}_metrics", {})
        for sub in ("precision", "recall", "f1"):
            bv = bclass.get(sub, 0)
            cv = cclass.get(sub, 0)
            rows.append({
                "metric": f"{cls.upper()} {sub.capitalize()}",
                "baseline": bv,
                "candidate": cv,
                "delta": _delta(bv, cv),
                "improved": (_delta(bv, cv) or 0) > 0,
            })

    comparison = {
        "baseline_tag": baseline.get("tag", "baseline"),
        "candidate_tag": candidate.get("tag", "candidate"),
        "baseline_provider": baseline.get("provider"),
        "candidate_provider": candidate.get("provider"),
        "baseline_samples": baseline.get("successful_predictions", 0),
        "candidate_samples": candidate.get("successful_predictions", 0),
        "rows": rows,
    }

    # Save comparison
    results_dir = Path(baseline_path).parent
    out_path = results_dir / f"comparison_{comparison['baseline_tag']}_vs_{comparison['candidate_tag']}.json"
    with open(out_path, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Comparison saved to {out_path}")

    return comparison
