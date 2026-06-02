"""Terminal report and optional charts for backtest results."""

from typing import Any


def _fmt_pct(v: float) -> str:
    return f"{v * 100:.1f}%" if isinstance(v, (int, float)) and v <= 1.0 else f"{v:.1f}%"


def _fmt_float(v: float) -> str:
    if v == float("inf"):
        return "∞"
    return f"{v:.2f}"


def print_report(result: dict[str, Any]) -> None:
    m = result.get("metrics", {})
    if not m:
        print("No metrics available.")
        return

    tag = result.get("tag", "?")
    provider = result.get("provider", "?")
    model = result.get("model", "?")

    try:
        from tabulate import tabulate  # noqa: F401

        _print_tabulate(result, m, tag, provider, model)
    except ImportError:
        _print_manual(result, m, tag, provider, model)


def _print_tabulate(result, m, tag, provider, model):
    from tabulate import tabulate

    print(f"\n{'=' * 60}")
    print(f"  BACKTEST REPORT — {tag}")
    print(f"  Provider: {provider} | Model: {model}")
    print(f"  Samples: {result.get('successful_predictions', 0)}/{result.get('total_samples', 0)}")
    print(f"  Elapsed: {result.get('elapsed_seconds', 0):.1f}s")
    print(f"{'=' * 60}")

    rows = [
        ["Direction Accuracy", _fmt_pct(m.get("direction_accuracy", 0))],
        ["Win Rate", _fmt_pct(m.get("win_rate", 0))],
        ["Profit Factor", _fmt_float(m.get("profit_factor", 0))],
        ["Avg Win", f"{m.get('avg_win', 0):.2f}%"],
        ["Avg Loss", f"{m.get('avg_loss', 0):.2f}%"],
        ["Sharpe Ratio", _fmt_float(m.get("sharpe_ratio", 0))],
        ["Max Drawdown", f"{m.get('max_drawdown', 0):.2f}%"],
    ]

    lm = m.get("long_metrics", {})
    sm = m.get("short_metrics", {})
    rows.extend(
        [
            ["", ""],
            ["LONG Precision", _fmt_pct(lm.get("precision", 0))],
            ["LONG Recall", _fmt_pct(lm.get("recall", 0))],
            ["LONG F1", _fmt_float(lm.get("f1", 0))],
            ["SHORT Precision", _fmt_pct(sm.get("precision", 0))],
            ["SHORT Recall", _fmt_pct(sm.get("recall", 0))],
            ["SHORT F1", _fmt_float(sm.get("f1", 0))],
        ]
    )

    print(tabulate(rows, headers=["Metric", "Value"], tablefmt="simple"))
    print()


def _print_manual(result, m, tag, provider, model):
    print(f"\n{'=' * 50}")
    print(f"  BACKTEST REPORT — {tag}")
    print(f"  Provider: {provider} | Model: {model}")
    print(f"  Samples: {result.get('successful_predictions', 0)}/{result.get('total_samples', 0)}")
    print(f"{'=' * 50}")
    print(f"  {'Metric':<25} {'Value':>10}")
    print(f"  {'-' * 36}")
    print(f"  {'Direction Accuracy':<25} {_fmt_pct(m.get('direction_accuracy', 0)):>10}")
    print(f"  {'Win Rate':<25} {_fmt_pct(m.get('win_rate', 0)):>10}")
    print(f"  {'Profit Factor':<25} {_fmt_float(m.get('profit_factor', 0)):>10}")
    print(f"  {'Avg Win':<25} {m.get('avg_win', 0):>9.2f}%")
    print(f"  {'Avg Loss':<25} {m.get('avg_loss', 0):>9.2f}%")
    print(f"  {'Sharpe Ratio':<25} {_fmt_float(m.get('sharpe_ratio', 0)):>10}")
    print(f"  {'Max Drawdown':<25} {m.get('max_drawdown', 0):>9.2f}%")
    print()


def print_comparison(comparison: dict[str, Any]) -> None:
    bt = comparison.get("baseline_tag", "baseline")
    ct = comparison.get("candidate_tag", "candidate")

    try:
        from tabulate import tabulate

        rows = []
        ratio_metrics = {
            "Direction Accuracy",
            "Win Rate",
            "LONG Precision",
            "LONG Recall",
            "SHORT Precision",
            "SHORT Recall",
        }
        for r in comparison.get("rows", []):
            bv = r["baseline"]
            cv = r["candidate"]
            delta = r.get("delta")
            improved = r.get("improved")
            arrow = "✅" if improved else "❌" if improved is False else ""
            is_ratio = r["metric"] in ratio_metrics
            bstr = _fmt_pct(bv) if is_ratio else _fmt_float(bv)
            cstr = _fmt_pct(cv) if is_ratio else _fmt_float(cv)
            dstr = f"{delta:+.4f}" if delta is not None else ""
            rows.append([r["metric"], bstr, cstr, dstr, arrow])

        print(f"\n{'=' * 70}")
        print(f"  COMPARISON: {bt} vs {ct}")
        print(f"{'=' * 70}")
        print(tabulate(rows, headers=["Metric", bt, ct, "Delta", ""], tablefmt="simple"))
        print()
    except ImportError:
        print(f"\n  COMPARISON: {bt} vs {ct}")
        print(f"  {'Metric':<25} {bt:>12} {ct:>12} {'Delta':>10}")
        print(f"  {'-' * 60}")
        for r in comparison.get("rows", []):
            bv = f"{r['baseline']:.4f}" if isinstance(r["baseline"], float) else str(r["baseline"])
            cv = f"{r['candidate']:.4f}" if isinstance(r["candidate"], float) else str(r["candidate"])
            d = f"{r['delta']:+.4f}" if r.get("delta") is not None else ""
            print(f"  {r['metric']:<25} {bv:>12} {cv:>12} {d:>10}")
        print()


def plot_equity_curve(result: dict[str, Any], save_path: str = None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping charts")
        return

    trades = result.get("trade_results", [])
    if not trades:
        return

    equity = [100.0]
    for t in trades:
        equity.append(equity[-1] * (1 + t.get("pnl_pct", 0) / 100))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(equity, linewidth=1.5)
    ax.set_title(f"Equity Curve — {result.get('tag', '?')}")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity")
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.5)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Chart saved to {save_path}")
    else:
        plt.show()
