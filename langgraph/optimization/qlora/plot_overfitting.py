#!/usr/bin/env python3
"""Overfitting dashboard for a finished QLoRA run — graph it, don't assume it.

Reads a result JSON written by ``train_qlora.py`` (and, if present, the matching
``<tag>_loss_curve.json``) and renders ONE figure with the four checks from
``docs/GUIA_IMPLEMENTACION_FIX_QLORA.md`` Apéndice A:

  1. Train vs eval loss per step      → does eval loss diverge upward?
  2. Train / val / test accuracy + gap → is train >> test (memorization)?
  3. Model vs heuristic baseline vs    → does it beat a trivial rule, or just
     majority class                       the ~51% coin flip?
  4. Per-symbol test accuracy          → is it fragile on some assets, or did
                                          the test collapse to ONE symbol (the
                                          old leakage signature)?

CPU-only, no GPU/torch needed — runs anywhere the result JSON lands.

Usage:
    python optimization/qlora/plot_overfitting.py --result optimization/qlora/results/qlora_cloud.json
    python optimization/qlora/plot_overfitting.py --result .../qlora_cloud.json --out dashboard.png
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Majority-class accuracy on the dataset (1022 LONG / 978 SHORT ≈ 51.1%). The
# meaningful bar to clear is the heuristic baseline, not this — but it's a useful
# floor line: anything near it learned nothing.
MAJORITY_CLASS = 0.511


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _direction_acc(preds: list[dict], actuals: list[dict]) -> float:
    if not preds:
        return 0.0
    correct = sum(1 for p, a in zip(preds, actuals) if p.get("bias") == a.get("bias"))
    return correct / len(preds)


def _per_symbol_accuracy(result: dict) -> dict[str, tuple[float, int]]:
    """Test direction accuracy per symbol, from sample_keys ('SYMBOL@ts')."""
    preds = result.get("predictions", [])
    actuals = result.get("actuals", [])
    keys = result.get("sample_keys", [])
    buckets: dict[str, list[bool]] = defaultdict(list)
    for k, p, a in zip(keys, preds, actuals):
        symbol = k.split("@", 1)[0]
        buckets[symbol].append(p.get("bias") == a.get("bias"))
    return {s: (sum(v) / len(v), len(v)) for s, v in buckets.items() if v}


def _plot_loss(ax, loss_history: list[dict]):
    train = [(r["step"], r["loss"]) for r in loss_history if "loss" in r and "step" in r]
    evals = [(r["step"], r["eval_loss"]) for r in loss_history if "eval_loss" in r and "step" in r]
    if not train and not evals:
        ax.text(0.5, 0.5, "no loss history\n(<tag>_loss_curve.json missing)", ha="center", va="center")
        ax.set_axis_off()
        return
    if train:
        ax.plot(*zip(*train), color="#3498db", label="train_loss")
    if evals:
        ax.plot(*zip(*evals), color="#e74c3c", marker="o", markersize=3, label="eval_loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("1. Train vs eval loss\n(eval rising = overfitting)")
    ax.legend()
    ax.grid(True, alpha=0.3)


def _plot_gap(ax, result: dict):
    ov = result.get("overfitting", {}) or {}
    test_acc = ov.get("test_acc")
    if test_acc is None:
        test_acc = result.get("test_metrics", {}).get("accuracy", result.get("best_score", 0.0))
    rows = [("train", ov.get("train_acc")), ("val", ov.get("val_acc")), ("test", test_acc)]
    rows = [(n, v) for n, v in rows if v is not None]
    names = [n for n, _ in rows]
    vals = [v for _, v in rows]
    colors = ["#95a5a6", "#f39c12", "#2ecc71"][: len(rows)]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    gap = ov.get("gap")
    title = "2. Accuracy by split"
    if gap is not None:
        verdict = "OK (<5pp)" if gap < 0.05 else ("watch (5-10pp)" if gap < 0.10 else "OVERFIT (>10pp)")
        title += f"\ngap(train-test) = {gap:+.3f} → {verdict}"
    ax.set_title(title)
    ax.set_ylabel("Direction accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.axhline(MAJORITY_CLASS, color="red", linestyle="--", alpha=0.6, label=f"majority {MAJORITY_CLASS:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)


def _plot_baseline(ax, result: dict):
    test_acc = result.get("overfitting", {}).get("test_acc")
    if test_acc is None:
        test_acc = result.get("test_metrics", {}).get("accuracy", result.get("best_score", 0.0))
    baseline = result.get("baseline_metrics", {}).get("direction_accuracy")
    rows = [
        ("fine-tuned", test_acc, "#2ecc71"),
        ("heuristic", baseline, "#9b59b6"),
        ("majority", MAJORITY_CLASS, "#95a5a6"),
    ]
    rows = [(n, v, c) for n, v, c in rows if v is not None]
    names = [n for n, _, _ in rows]
    vals = [v for _, v, _ in rows]
    colors = [c for _, _, c in rows]
    bars = ax.bar(names, vals, color=colors)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
    edge = (test_acc - baseline) if baseline is not None else None
    title = "3. vs baselines"
    if edge is not None:
        title += f"\nedge over heuristic = {edge:+.3f}"
    ax.set_title(title)
    ax.set_ylabel("Direction accuracy")
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, axis="y", alpha=0.3)


def _plot_per_symbol(ax, result: dict):
    per_sym = _per_symbol_accuracy(result)
    if not per_sym:
        ax.text(0.5, 0.5, "no per-sample data", ha="center", va="center")
        ax.set_axis_off()
        return
    items = sorted(per_sym.items(), key=lambda kv: kv[1][0])
    names = [s for s, _ in items]
    vals = [acc for _, (acc, _) in items]
    counts = [n for _, (_, n) in items]
    bars = ax.barh(names, vals, color="#3498db")
    for b, v, c in zip(bars, vals, counts):
        ax.text(v + 0.01, b.get_y() + b.get_height() / 2, f"{v:.2f} (n={c})", va="center", fontsize=7)
    n_sym = len(names)
    title = f"4. Test accuracy by symbol ({n_sym} symbol{'s' if n_sym != 1 else ''})"
    if n_sym == 1:
        title += "\n⚠ ONE symbol = leakage signature"
    ax.set_title(title)
    ax.set_xlabel("Direction accuracy")
    ax.set_xlim(0.0, 1.05)
    ax.axvline(MAJORITY_CLASS, color="red", linestyle="--", alpha=0.6)
    ax.grid(True, axis="x", alpha=0.3)


def build_dashboard(result_path: Path, out_path: Path | None = None) -> Path:
    result = _load(result_path)
    tag = result.get("tag", result_path.stem)

    # Loss curve lives in the same per-run folder as result.json.
    loss_history: list[dict] = []
    loss_path = result_path.parent / "loss_curve.json"
    if loss_path.exists():
        loss_history = _load(loss_path)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    _plot_loss(axes[0][0], loss_history)
    _plot_gap(axes[0][1], result)
    _plot_baseline(axes[1][0], result)
    _plot_per_symbol(axes[1][1], result)
    fig.suptitle(f"QLoRA overfitting dashboard — {tag}", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    out = out_path or result_path.parent / f"{tag}_overfitting.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main():
    parser = argparse.ArgumentParser(description="Render the QLoRA overfitting dashboard from a result JSON")
    parser.add_argument("--result", required=True, help="Path to qlora_<tag>.json")
    parser.add_argument("--out", help="Output PNG path (default: <tag>_overfitting.png next to the result)")
    args = parser.parse_args()

    out = build_dashboard(Path(args.result), Path(args.out) if args.out else None)
    print(f"Dashboard saved: {out}")


if __name__ == "__main__":
    main()
