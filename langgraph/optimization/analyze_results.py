#!/usr/bin/env python3
"""Comparación de resultados de optimización entre todos los modelos.

Uso:
    python analyze_results.py
    python analyze_results.py --results-dir /ruta/a/results
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def load_results(results_dir: Path) -> dict:
    """Carga todos los archivos *_optimization.json."""
    data = {}
    for f in sorted(results_dir.glob("*_optimization.json")):
        model = f.stem.replace("_optimization", "")
        with open(f) as fh:
            data[model] = json.load(fh)
        log.info(f"Cargado {model}: best_score={data[model].get('best_score')}")
    return data


def plot_comparison(data: dict, output: Path):
    """Gráfica de barras con mejores scores + heatmaps para modelos con grids 2D."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models_with_scores = {k: v for k, v in data.items() if v.get("best_score") is not None}
    if not models_with_scores:
        log.warning("No hay modelos con scores para graficar.")
        return

    n_heatmaps = sum(1 for v in models_with_scores.values() if len(v.get("all_results", [])) > 1)
    n_plots = 1 + n_heatmaps
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # --- Gráfica de barras ---
    names = list(models_with_scores.keys())
    scores = [models_with_scores[n]["best_score"] for n in names]
    colors = ["#2ecc71" if s == max(scores) else "#3498db" for s in scores]

    ax = axes[0]
    bars = ax.bar(names, scores, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_ylabel("Mejor Accuracy")
    ax.set_title("Comparación de Modelos — Mejores Scores")
    ax.set_ylim(min(scores) - 0.05, max(scores) + 0.05)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{score:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # --- Heatmaps para modelos con grids 2D ---
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
    fig.savefig(output, dpi=150)
    plt.close(fig)
    log.info(f"Gráfica de comparación guardada en {output}")


def print_summary(data: dict):
    """Imprime tabla comparativa."""
    print(f"\n{'='*70}")
    print(f"  Resumen Comparativo de Modelos")
    print(f"{'='*70}")
    print(f"  {'Modelo':<18} {'Mejor Score':<12} {'Método':<10} {'Fits':<8} {'Tiempo':<10}")
    print(f"  {'-'*58}")
    for name, result in sorted(data.items(), key=lambda x: x[1].get("best_score") or 0, reverse=True):
        score = result.get("best_score")
        score_str = f"{score:.4f}" if score is not None else "N/A"
        method = result.get("search_method", "?")
        fits = result.get("total_fits", 0)
        elapsed = result.get("elapsed_seconds", 0)
        m, s = divmod(int(elapsed), 60)
        time_str = f"{m}m {s}s" if elapsed else "N/A"
        print(f"  {name:<18} {score_str:<12} {method:<10} {fits:<8} {time_str:<10}")

    best = max((v for v in data.values() if v.get("best_score")), key=lambda x: x["best_score"], default=None)
    if best:
        print(f"\n  🏆 Mejor modelo: {best['model']} ({best['best_score']:.4f})")
        print(f"     Params: {json.dumps(best['best_params'], indent=6)}")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description="Analizar resultados de optimización")
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    data = load_results(results_dir)

    if not data:
        log.error(f"No se encontraron archivos *_optimization.json en {results_dir}")
        return

    print_summary(data)
    plot_comparison(data, results_dir / "comparison.png")


if __name__ == "__main__":
    main()
