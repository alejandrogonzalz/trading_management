"""CLI entry point for the backtest framework."""

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.config import DEFAULT_MONTHS, DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES  # noqa: E402
from backtest.evaluation.compare import compare  # noqa: E402
from backtest.evaluation.report import print_comparison, print_report  # noqa: E402
from backtest.evaluation.runner import LLMBacktestRunner, MLBacktestRunner  # noqa: E402
from backtest.export import export_training_data  # noqa: E402
from backtest.pipeline import LABELED_DIR, DataPipeline  # noqa: E402

_DEFAULT_SYMBOLS_STR = ",".join(DEFAULT_SYMBOLS)
_DEFAULT_TIMEFRAMES_STR = ",".join(DEFAULT_TIMEFRAMES)


def cmd_fetch_candles(args):
    symbols = [s.strip() for s in args.symbols.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")]

    pipe = DataPipeline(
        symbols=symbols,
        timeframes=timeframes,
        months=args.months,
        base_tf=args.interval if args.interval in timeframes else timeframes[0],
    )
    asyncio.run(pipe.fetch(skip_existing=False))


def cmd_prepare_dataset(args):
    symbols = [s.strip() for s in args.symbols.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")]
    base_tf = args.interval if args.interval in timeframes else "1h"

    output_path = Path(args.output) if args.output else None
    apply_drawdown_filter = not args.no_drawdown_filter
    if not apply_drawdown_filter:
        print("Drawdown-before-profit filter DISABLED (no-drawdown-filter ablation).")
        if output_path is None:
            # Never silently overwrite the production dataset.jsonl with an
            # unfiltered build — default to a clearly-named sibling file.
            output_path = LABELED_DIR / "dataset_no_drawdown_filter.jsonl"
            print(f"  No --output given; writing to {output_path}")

    pipe = DataPipeline(
        symbols=symbols,
        timeframes=timeframes,
        base_tf=base_tf,
        output_path=output_path,
        apply_drawdown_filter=apply_drawdown_filter,
    )
    pipe.build_dataset()


def cmd_run_backtest(args):
    split = args.split if args.split != "none" else None
    runner = LLMBacktestRunner(
        dataset_path=args.dataset,
        tag=args.tag,
        provider=args.provider,
        model=args.model,
        max_samples=args.max_samples,
        verbose=args.verbose,
        split=split,
    )
    result = asyncio.run(runner.run())
    print_report(result)


def cmd_compare(args):
    comparison = compare(args.baseline, args.candidate)
    print_comparison(comparison)


def cmd_export_training_data(args):
    export_training_data(
        dataset_path=args.dataset,
        output_dir=args.output,
        mode=args.mode,
    )


def cmd_compare_stats(args):
    from optimization.stats_tests import compare as compare_stats

    compare_stats(args.a, args.b)


def cmd_train_ml(args):
    runner = MLBacktestRunner(
        dataset_path=args.dataset,
        model_type=args.model,
        tag=args.tag,
        max_samples=args.max_samples,
        verbose=args.verbose,
        serialize=args.serialize,
    )
    result = runner.run()
    print_report(result)


def main():
    parser = argparse.ArgumentParser(description="Backtest Framework CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # fetch-candles
    p = sub.add_parser("fetch-candles", help="Download historical candles from Binance")
    p.add_argument("--symbols", default=_DEFAULT_SYMBOLS_STR)
    p.add_argument("--interval", default="1h")
    p.add_argument("--months", type=int, default=DEFAULT_MONTHS)
    p.add_argument("--timeframes", default=_DEFAULT_TIMEFRAMES_STR)

    # prepare-dataset
    p = sub.add_parser("prepare-dataset", help="Calculate indicators and generate labeled dataset")
    p.add_argument("--symbols", default=_DEFAULT_SYMBOLS_STR)
    p.add_argument("--interval", default="1h")
    p.add_argument("--timeframes", default=_DEFAULT_TIMEFRAMES_STR)
    p.add_argument("--output", help="Output JSONL path (default: data/labeled/dataset.jsonl)")
    p.add_argument(
        "--no-drawdown-filter",
        action="store_true",
        help=(
            "Skip the labeler's drawdown-before-profit survivorship filter "
            "(experiment/no-drawdown-filter ablation). Defaults output to "
            "data/labeled/dataset_no_drawdown_filter.jsonl so the production "
            "dataset is never overwritten. See docs/AUDIT_QLORA_88PCT.md §2."
        ),
    )

    # run-backtest
    p = sub.add_parser("run-backtest", help="Run backtest with LLM predictions")
    p.add_argument("--dataset", required=True)
    p.add_argument("--provider", help="LLM provider (groq, deepseek, mock, etc.)")
    p.add_argument("--model", help="LLM model name")
    p.add_argument("--tag", default="backtest")
    p.add_argument("--max-samples", type=int)
    p.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "none"],
        help="Dataset split to evaluate on (default: test). Use 'none' to run on the full dataset.",
    )
    p.add_argument("--verbose", "-v", action="store_true")

    # compare
    p = sub.add_parser("compare", help="Compare two backtest runs")
    p.add_argument("--baseline", required=True)
    p.add_argument("--candidate", required=True)

    # compare-stats
    p = sub.add_parser("compare-stats", help="McNemar + paired t-test between two result JSONs")
    p.add_argument("--a", required=True, help="Result JSON for model A")
    p.add_argument("--b", required=True, help="Result JSON for model B")

    # export-training-data
    p = sub.add_parser("export-training-data", help="Export labeled dataset as chat-format JSONL for fine-tuning")
    p.add_argument("--dataset", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--mode", default="FUTURES", choices=["SPOT", "FUTURES"])

    # train-ml
    p = sub.add_parser("train-ml", help="Train and backtest a traditional ML model")
    p.add_argument("--dataset", required=True)
    p.add_argument("--model", default="xgboost", choices=["xgboost", "random-forest", "lstm"])
    p.add_argument("--tag")
    p.add_argument("--max-samples", type=int)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--serialize", action="store_true", help="Save trained model after evaluation")

    args = parser.parse_args()

    commands = {
        "fetch-candles": cmd_fetch_candles,
        "prepare-dataset": cmd_prepare_dataset,
        "run-backtest": cmd_run_backtest,
        "compare": cmd_compare,
        "compare-stats": cmd_compare_stats,
        "export-training-data": cmd_export_training_data,
        "train-ml": cmd_train_ml,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
