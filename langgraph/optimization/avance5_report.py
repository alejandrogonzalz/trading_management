#!/usr/bin/env python3
"""Avance 5 — Ensemble Models: results report and comparison.

Loads the JSON results produced by run_ensembles.py and the individual-model
results from Avance 4 (optimization/results/lstm_optimization.json, etc.)
then prints a full comparison table and saves charts to optimization/results/.

Usage:
    cd langgraph/
    python optimization/avance5_report.py
    python optimization/avance5_report.py --results-dir optimization/results
    python optimization/avance5_report.py --no-plots        # skip matplotlib
"""

import argparse
import json
import sys
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ── Ensemble architecture descriptions ──────────────────────────────────────

ENSEMBLE_DESCRIPTIONS = {
    "bagging_lstm": {
        "type": "Bagging",
        "full_name": "Bagging-LSTM",
        "description": (
            "Trains N independent LSTMs with different random seeds on the full "
            "train+val set.  Combines predictions via soft-voting (mean of class "
            "probabilities).  Reduces variance without increasing bias."
        ),
        "key_params": ["n_bags", "lstm hidden_size", "lstm num_layers", "lstm sequence_length"],
    },
    "adaboost": {
        "type": "Boosting",
        "full_name": "AdaBoost",
        "description": (
            "Sequential boosting over shallow decision-tree stumps (depth=1). "
            "Each round re-weights samples that were mis-classified by the "
            "previous round.  Grid-searched over n_estimators and learning_rate."
        ),
        "key_params": ["n_estimators", "learning_rate"],
    },
    "soft_voting": {
        "type": "Voting",
        "full_name": "Soft-Voting Ensemble",
        "description": (
            "Combines three heterogeneous models (LSTM, XGBoost, Random Forest) "
            "via a weighted average of their class probabilities.  Weights are "
            "proportional to each model's validation accuracy, so stronger "
            "models get more influence."
        ),
        "key_params": ["lstm weight", "xgb weight", "rf weight"],
    },
    "stacking": {
        "type": "Stacking",
        "full_name": "Stacking (OOF)",
        "description": (
            "Two-level meta-learner.  Base models (LSTM, XGBoost, RF) produce "
            "out-of-fold probability predictions on the training set via "
            "TimeSeriesSplit(5).  A Logistic Regression meta-learner is then "
            "trained on these OOF features.  Final predictions use base models "
            "trained on all data + the meta-learner."
        ),
        "key_params": ["cv_splits (OOF)", "meta-learner (LogisticRegression)"],
    },
    "blending": {
        "type": "Blending",
        "full_name": "Blending",
        "description": (
            "Similar to stacking but simpler: a hold-out blend set (30% of "
            "training data, kept out-of-sample) is used to train the "
            "Logistic Regression meta-learner on base-model probability outputs. "
            "No cross-validation needed; faster but uses less training data."
        ),
        "key_params": ["blend_fraction (0.30)", "meta-learner (LogisticRegression)"],
    },
}

AVANCE4_MODELS = ["lstm", "xgboost", "random_forest", "svm", "adaboost", "logistic_regression", "knn"]


# ── Loader ───────────────────────────────────────────────────────────────────

def load_results(results_dir: Path) -> tuple[dict, dict]:
    """Return (avance4_results, avance5_results) dicts keyed by model name."""
    all_files = list(results_dir.glob("*_optimization.json"))
    a4, a5 = {}, {}
    for f in sorted(all_files):
        name = f.stem.replace("_optimization", "")
        data = json.loads(f.read_text())
        if name in AVANCE4_MODELS:
            a4[name] = data
        else:
            a5[name] = data
    return a4, a5


# ── Metric extraction ────────────────────────────────────────────────────────

def _get_metrics(result: dict, split: str = "test") -> dict:
    """Extract accuracy/f1/auc from a result dict (handles both schemas)."""
    key = f"{split}_metrics"
    m = result.get(key) or {}
    if not m:
        # Avance-4 individual models store best_score as val accuracy
        if split == "val":
            m = {"accuracy": result.get("best_score")}
        else:
            m = {}
    return {
        "accuracy": m.get("accuracy"),
        "f1_macro": m.get("f1_macro"),
        "auc_roc": m.get("auc_roc"),
    }


def _fmt(v, fmt=".4f"):
    return format(v, fmt) if v is not None else "  N/A "


# ── Print functions ──────────────────────────────────────────────────────────

def print_ensemble_architectures(a5: dict):
    print(f"\n{'═' * 72}")
    print("  ENSEMBLE ARCHITECTURES — Avance 5")
    print(f"{'═' * 72}")
    for name, desc in ENSEMBLE_DESCRIPTIONS.items():
        result = a5.get(name, {})
        bp = result.get("best_params", {})
        elapsed = result.get("elapsed_seconds", 0)
        m_s, m_r = divmod(int(elapsed), 60)

        print(f"\n  ┌─ {desc['full_name']} ({desc['type']})")
        print(f"  │  {desc['description']}")
        if bp:
            print(f"  │  Best params: {json.dumps(bp, indent=0)}")
        else:
            print(f"  │  Key params: {', '.join(desc['key_params'])}")
        if elapsed:
            print(f"  │  Training time: {m_s}m {m_r}s")
        print(f"  └{'─' * 68}")


def print_comparison_table(a4: dict, a5: dict):
    print(f"\n{'═' * 78}")
    print("  FULL MODEL COMPARISON  (Avance 4 individual  vs  Avance 5 ensembles)")
    print(f"{'═' * 78}")
    header = f"  {'Model':<24} {'Type':<12} {'Val Acc':>8} {'Test Acc':>9} {'F1-macro':>9} {'AUC-ROC':>8} {'Time':>7}"
    print(header)
    print(f"  {'─' * 74}")

    rows = []

    for name, result in a4.items():
        vm = _get_metrics(result, "val")
        tm = _get_metrics(result, "test")
        elapsed = result.get("elapsed_seconds", 0)
        rows.append({
            "name": name, "group": "A4",
            "val_acc": vm["accuracy"], "test_acc": tm["accuracy"],
            "f1": tm["f1_macro"], "auc": tm["auc_roc"],
            "elapsed": elapsed,
        })

    for name, result in a5.items():
        vm = _get_metrics(result, "val")
        tm = _get_metrics(result, "test")
        elapsed = result.get("elapsed_seconds", 0)
        rows.append({
            "name": name, "group": "A5",
            "val_acc": vm["accuracy"], "test_acc": tm["accuracy"],
            "f1": tm["f1_macro"], "auc": tm["auc_roc"],
            "elapsed": elapsed,
        })

    rows.sort(key=lambda r: r["test_acc"] or r["val_acc"] or 0, reverse=True)

    prev_group = None
    for r in rows:
        if r["group"] != prev_group:
            label = "── Avance 4 ─────" if r["group"] == "A4" else "── Avance 5 (ensembles) ─"
            print(f"\n  {label}")
            prev_group = r["group"]

        desc = ENSEMBLE_DESCRIPTIONS.get(r["name"], {})
        etype = desc.get("type", "—")
        m, s = divmod(int(r["elapsed"] or 0), 60)
        tstr = f"{m}m{s:02d}s" if r["elapsed"] else "  N/A"
        print(
            f"  {r['name']:<24} {etype:<12} "
            f"{_fmt(r['val_acc']):>8} {_fmt(r['test_acc']):>9} "
            f"{_fmt(r['f1']):>9} {_fmt(r['auc']):>8} {tstr:>7}"
        )

    # Best overall
    best = max(rows, key=lambda r: r["test_acc"] or r["val_acc"] or 0)
    print(f"\n  Best overall: {best['name']} — test acc {_fmt(best['test_acc'])}")

    # Best ensemble
    a5_rows = [r for r in rows if r["group"] == "A5"]
    if a5_rows:
        best_ens = max(a5_rows, key=lambda r: r["test_acc"] or r["val_acc"] or 0)
        print(f"  Best ensemble: {best_ens['name']} — test acc {_fmt(best_ens['test_acc'])}")

    # Improvement over LSTM (Avance 4 winner)
    lstm_row = next((r for r in rows if r["name"] == "lstm"), None)
    if lstm_row and a5_rows:
        best_ens = max(a5_rows, key=lambda r: r["test_acc"] or r["val_acc"] or 0)
        if lstm_row["test_acc"] and best_ens["test_acc"]:
            delta = (best_ens["test_acc"] - lstm_row["test_acc"]) * 100
            sign = "+" if delta >= 0 else ""
            print(f"  Δ vs LSTM (Avance 4 winner): {sign}{delta:.2f} pp")

    print(f"\n{'═' * 78}")


def print_best_params_detail(a5: dict):
    print(f"\n{'═' * 72}")
    print("  BEST HYPERPARAMETERS — Ensemble Models")
    print(f"{'═' * 72}")
    for name, result in a5.items():
        bp = result.get("best_params", {})
        score = result.get("best_score")
        vm = _get_metrics(result, "val")
        tm = _get_metrics(result, "test")
        desc = ENSEMBLE_DESCRIPTIONS.get(name, {})
        print(f"\n  {desc.get('full_name', name)}")
        print(f"  {'─' * 50}")
        if score:
            print(f"    best_score (val):  {score:.4f}")
        if vm.get("accuracy"):
            print(f"    val  → acc={vm['accuracy']:.4f}  f1={_fmt(vm['f1_macro'])}  auc={_fmt(vm['auc_roc'])}")
        if tm.get("accuracy"):
            print(f"    test → acc={tm['accuracy']:.4f}  f1={_fmt(tm['f1_macro'])}  auc={_fmt(tm['auc_roc'])}")
        if bp:
            for k, v in bp.items():
                print(f"    {k}: {v}")


# ── Plots ────────────────────────────────────────────────────────────────────

def save_plots(a4: dict, a5: dict, out_dir: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("  (matplotlib not available — skipping plots)")
        return

    all_models = {}
    for name, r in {**a4, **a5}.items():
        vm = _get_metrics(r, "val")
        tm = _get_metrics(r, "test")
        all_models[name] = {
            "val_acc": vm.get("accuracy"),
            "test_acc": tm.get("accuracy"),
            "f1": tm.get("f1_macro"),
            "auc": tm.get("auc_roc"),
            "is_ensemble": name in a5,
        }

    names = list(all_models.keys())
    test_accs = [all_models[n]["test_acc"] or all_models[n]["val_acc"] or 0 for n in names]
    colors = ["#e74c3c" if all_models[n]["is_ensemble"] else "#3498db" for n in names]

    # Sort by accuracy
    order = sorted(range(len(names)), key=lambda i: test_accs[i], reverse=True)
    names = [names[i] for i in order]
    test_accs = [test_accs[i] for i in order]
    colors = [colors[i] for i in order]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Avance 5 — Ensemble vs Individual Models", fontsize=14, fontweight="bold")

    # ── Plot 1: test accuracy bar chart
    ax = axes[0]
    bars = ax.bar(range(len(names)), test_accs, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy (test set)")
    ax.set_title("Test Accuracy — All Models")
    ax.set_ylim(max(0, min(test_accs) - 0.05), min(1.0, max(test_accs) + 0.05))
    for bar, acc in zip(bars, test_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                f"{acc:.3f}", ha="center", va="bottom", fontsize=7)

    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(color="#3498db", label="Avance 4 (individual)"),
        Patch(color="#e74c3c", label="Avance 5 (ensemble)"),
    ], fontsize=8)

    # ── Plot 2: F1 + AUC for ensembles only
    ax = axes[1]
    ens_names = [n for n in names if all_models[n]["is_ensemble"]]
    f1s = [all_models[n]["f1"] or 0 for n in ens_names]
    aucs = [all_models[n]["auc"] or 0 for n in ens_names]
    x = np.arange(len(ens_names))
    w = 0.35
    ax.bar(x - w / 2, f1s, w, label="F1-macro", color="#e74c3c", alpha=0.85)
    ax.bar(x + w / 2, aucs, w, label="AUC-ROC", color="#c0392b", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ens_names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Ensemble Models — F1 & AUC")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)

    # ── Plot 3: radar / grouped bar — val vs test for top ensembles
    ax = axes[2]
    ens_val = [all_models[n]["val_acc"] or 0 for n in ens_names]
    ens_test = [all_models[n]["test_acc"] or 0 for n in ens_names]
    ax.bar(x - w / 2, ens_val, w, label="Val accuracy", color="#f39c12", alpha=0.85)
    ax.bar(x + w / 2, ens_test, w, label="Test accuracy", color="#e74c3c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(ens_names, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Accuracy")
    ax.set_title("Ensemble Models — Val vs Test")
    ax.set_ylim(max(0, min(ens_val + ens_test) - 0.05), 1.0)
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = out_dir / "avance5_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"\n  Chart saved → {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Avance 5 results report")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR), help="Path to optimization/results/")
    parser.add_argument("--no-plots", action="store_true", help="Skip matplotlib charts")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"ERROR: results dir not found: {results_dir}", file=sys.stderr)
        sys.exit(1)

    a4, a5 = load_results(results_dir)

    if not a5:
        print(
            f"No ensemble result files found in {results_dir}.\n"
            "Run  python optimization/run_ensembles.py  first.",
            file=sys.stderr,
        )
        sys.exit(1)

    print_ensemble_architectures(a5)
    print_comparison_table(a4, a5)
    print_best_params_detail(a5)

    if not args.no_plots:
        save_plots(a4, a5, results_dir)

    print("\nDone.\n")


if __name__ == "__main__":
    main()
