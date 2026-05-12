"""Metric calculations for backtest results."""

import math
from typing import List, Dict, Any
from collections import defaultdict


def direction_accuracy(predictions: List[Dict], actuals: List[Dict]) -> float:
    if not predictions:
        return 0.0
    correct = sum(1 for p, a in zip(predictions, actuals) if p["bias"] == a["bias"])
    return correct / len(predictions)


def precision_recall_f1(predictions: List[Dict], actuals: List[Dict], cls: str) -> Dict[str, float]:
    tp = sum(1 for p, a in zip(predictions, actuals) if p["bias"] == cls and a["bias"] == cls)
    fp = sum(1 for p, a in zip(predictions, actuals) if p["bias"] == cls and a["bias"] != cls)
    fn = sum(1 for p, a in zip(predictions, actuals) if p["bias"] != cls and a["bias"] == cls)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def win_rate(results: List[Dict]) -> float:
    if not results:
        return 0.0
    wins = sum(1 for r in results if r.get("outcome") == "WIN")
    return wins / len(results)


def profit_factor(results: List[Dict]) -> float:
    gross_profit = sum(r["pnl_pct"] for r in results if r.get("pnl_pct", 0) > 0)
    gross_loss = abs(sum(r["pnl_pct"] for r in results if r.get("pnl_pct", 0) < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return gross_profit / gross_loss


def avg_win_loss(results: List[Dict]) -> Dict[str, float]:
    wins = [r["pnl_pct"] for r in results if r.get("pnl_pct", 0) > 0]
    losses = [r["pnl_pct"] for r in results if r.get("pnl_pct", 0) < 0]
    return {
        "avg_win": sum(wins) / len(wins) if wins else 0.0,
        "avg_loss": sum(losses) / len(losses) if losses else 0.0,
    }


def sharpe_ratio(results: List[Dict], annualize_factor: float = 365.0) -> float:
    returns = [r.get("pnl_pct", 0) for r in results]
    if len(returns) < 2:
        return 0.0
    mean_r = sum(returns) / len(returns)
    std_r = (sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)) ** 0.5
    if std_r == 0:
        return 0.0
    return (mean_r / std_r) * math.sqrt(annualize_factor)


def max_drawdown(results: List[Dict]) -> float:
    if not results:
        return 0.0
    equity = 100.0
    peak = equity
    max_dd = 0.0
    for r in results:
        equity *= 1 + r.get("pnl_pct", 0) / 100
        peak = max(peak, equity)
        dd = (peak - equity) / peak * 100
        max_dd = max(max_dd, dd)
    return max_dd


def confidence_calibration(predictions: List[Dict], results: List[Dict], bins: int = 5) -> List[Dict]:
    buckets: Dict[int, List[bool]] = defaultdict(list)
    for p, r in zip(predictions, results):
        conf = p.get("confidence", 5)
        bucket = min(conf // (10 // bins), bins - 1)
        buckets[bucket].append(r.get("outcome") == "WIN")

    calibration = []
    for b in range(bins):
        items = buckets.get(b, [])
        lo = b * (10 // bins)
        hi = lo + (10 // bins)
        calibration.append({
            "confidence_range": f"{lo}-{hi}",
            "count": len(items),
            "actual_accuracy": sum(items) / len(items) if items else 0.0,
        })
    return calibration


def compute_all_metrics(predictions: List[Dict], actuals: List[Dict], results: List[Dict]) -> Dict[str, Any]:
    wl = avg_win_loss(results)
    return {
        "total_samples": len(predictions),
        "direction_accuracy": direction_accuracy(predictions, actuals),
        "long_metrics": precision_recall_f1(predictions, actuals, "LONG"),
        "short_metrics": precision_recall_f1(predictions, actuals, "SHORT"),
        "win_rate": win_rate(results),
        "profit_factor": profit_factor(results),
        "avg_win": wl["avg_win"],
        "avg_loss": wl["avg_loss"],
        "sharpe_ratio": sharpe_ratio(results),
        "max_drawdown": max_drawdown(results),
        "confidence_calibration": confidence_calibration(predictions, results),
    }
