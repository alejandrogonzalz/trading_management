import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


def save_optimization_plot(result: dict, output_path: Path):
    """Horizontal bar chart of top-30 configurations for a single model."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_res = result.get("all_results", [])
    if not all_res or all_res[0].get("mean_score") is None:
        log.info("No plottable results, skipping chart.")
        return

    scores = [r["mean_score"] for r in all_res[:30]]
    labels = [f"#{i + 1}" for i in range(len(scores))]
    colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(scores))]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(labels[::-1], scores[::-1], color=colors[::-1])
    ax.set_xlabel("Accuracy")
    ax.set_title(f"{result['model'].upper()} — Top {len(scores)} Configurations")
    best = result.get("best_score")
    if best:
        ax.axvline(x=best, color="red", linestyle="--", alpha=0.7, label=f"Best: {best:.4f}")
        ax.legend()
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Plot saved to {out}")


def save_loss_curve_plot(log_history: list[dict], output_path: Path, title: str | None = None):
    """Plot train vs eval loss per step from a HF ``trainer.state.log_history``.

    The overfitting tell is eval loss rising while train loss keeps falling.
    Train-loss records carry ``loss``; eval records carry ``eval_loss`` (both with
    ``step``). Skips cleanly if neither series is present.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train = [(r["step"], r["loss"]) for r in log_history if "loss" in r and "step" in r]
    evals = [(r["step"], r["eval_loss"]) for r in log_history if "eval_loss" in r and "step" in r]
    if not train and not evals:
        log.info("No loss history to plot, skipping loss curve.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    if train:
        ax.plot(*zip(*train), color="#3498db", label="train_loss")
    if evals:
        ax.plot(*zip(*evals), color="#e74c3c", marker="o", markersize=3, label="eval_loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title or "Training vs Eval Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Loss curve saved to {out}")


def save_comparison_plot(data: dict, output: Path):
    """Bar chart comparing best scores across all models, plus heatmaps for 2D grids."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models_with_scores = {k: v for k, v in data.items() if v.get("best_score") is not None}
    if not models_with_scores:
        log.warning("No models with scores to plot.")
        return

    n_heatmaps = sum(1 for v in models_with_scores.values() if len(v.get("all_results", [])) > 1)
    n_plots = 1 + n_heatmaps
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    names = list(models_with_scores.keys())
    scores = [models_with_scores[n]["best_score"] for n in names]
    colors = ["#2ecc71" if s == max(scores) else "#3498db" for s in scores]

    ax = axes[0]
    bars = ax.bar(names, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Best Accuracy")
    ax.set_title("Model Comparison — Best Scores")
    ax.set_ylim(min(scores) - 0.05, max(scores) + 0.05)
    for bar, score in zip(bars, scores):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    ax_idx = 1
    for model_name, result in models_with_scores.items():
        all_res = result.get("all_results", [])
        if len(all_res) <= 1:
            continue
        param_keys = list(all_res[0]["params"].keys())
        if len(param_keys) < 2:
            continue

        param_unique = {k: len(set(str(r["params"].get(k)) for r in all_res)) for k in param_keys}
        top2 = sorted(param_unique, key=param_unique.get, reverse=True)[:2]
        p1, p2 = top2
        vals1 = sorted(set(r["params"][p1] for r in all_res))
        vals2 = sorted(set(r["params"][p2] for r in all_res))

        grid = np.full((len(vals1), len(vals2)), np.nan)
        for r in all_res:
            v1, v2 = r["params"][p1], r["params"][p2]
            if v1 in vals1 and v2 in vals2:
                i, j = vals1.index(v1), vals2.index(v2)
                if np.isnan(grid[i, j]) or r["mean_score"] > grid[i, j]:
                    grid[i, j] = r["mean_score"]

        if np.all(np.isnan(grid)):
            continue

        ax = axes[ax_idx]
        im = ax.imshow(grid, cmap="YlGn", aspect="auto")
        ax.set_xticks(range(len(vals2)))
        ax.set_xticklabels([str(v) for v in vals2], fontsize=8)
        ax.set_yticks(range(len(vals1)))
        ax.set_yticklabels([str(v) for v in vals1], fontsize=8)
        ax.set_xlabel(p2)
        ax.set_ylabel(p1)
        ax.set_title(f"{model_name} — {p1} vs {p2}")
        fig.colorbar(im, ax=ax, shrink=0.8)
        for i in range(len(vals1)):
            for j in range(len(vals2)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", fontsize=7)
        ax_idx += 1

    plt.tight_layout()
    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    log.info(f"Comparison plot saved to {out}")
