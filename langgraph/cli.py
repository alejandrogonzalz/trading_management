"""CLI entry point for the backtest framework."""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Ensure the langgraph dir is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from backtest.fetch_candles import fetch_multi_tf_candles, save_candles  # noqa: E402
from backtest.calculate_indicators import calculate_indicators_batch, calculate_multi_tf_indicators  # noqa: E402
from backtest.label_data import generate_labeled_dataset, save_labeled_dataset  # noqa: E402
from backtest.run_backtest import run_backtest  # noqa: E402
from backtest.compare import compare  # noqa: E402
from backtest.report import print_report, print_comparison  # noqa: E402
from backtest.export_training_data import export_training_data  # noqa: E402


def cmd_fetch_candles(args):
    symbols = [s.strip() for s in args.symbols.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.months * 30)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    async def _fetch():
        for sym in symbols:
            print(f"Fetching {sym} timeframes={timeframes} from {start_str} to {end_str}...")
            candles_by_tf = await fetch_multi_tf_candles(sym, timeframes, start_str, end_str)
            for tf, candles in candles_by_tf.items():
                path = save_candles(candles, sym, tf)
                print(f"  Saved {len(candles)} {tf} candles to {path}")

    asyncio.run(_fetch())


def cmd_prepare_dataset(args):
    symbols = [s.strip() for s in args.symbols.split(",")]
    timeframes = [t.strip() for t in args.timeframes.split(",")]
    candles_dir = Path(__file__).parent / "backtest" / "data" / "candles"
    labeled_dir = Path(__file__).parent / "backtest" / "data" / "labeled"
    base_tf = args.interval

    all_labeled = []
    for sym in symbols:
        # Check which TF files exist
        available_tfs = {}
        for tf in timeframes:
            candle_file = candles_dir / f"{sym}_{tf}.json"
            if candle_file.exists():
                with open(candle_file) as f:
                    available_tfs[tf] = json.load(f)

        if base_tf not in available_tfs:
            print(f"No candle data for {sym}_{base_tf}. Run fetch-candles first.")
            continue

        use_multi_tf = len(available_tfs) > 1
        print(f"{'Multi-TF' if use_multi_tf else 'Single-TF'} mode for {sym}: {list(available_tfs.keys())}")

        if use_multi_tf:
            print(f"Calculating multi-TF indicators for {sym}...")
            indicators = calculate_multi_tf_indicators(available_tfs, base_tf=base_tf)
        else:
            candles = available_tfs[base_tf]
            print(f"Calculating indicators for {sym} ({len(candles)} candles)...")
            indicators = calculate_indicators_batch(candles)

        print(f"  Got {len(indicators)} indicator points")

        print(f"Labeling {sym}...")
        labeled = generate_labeled_dataset(available_tfs[base_tf], indicators, sym)
        print(f"  Got {len(labeled)} labeled samples (LONG/SHORT only, ambiguous discarded)")

        longs = sum(1 for s in labeled if s["label"]["bias"] == "LONG")
        shorts = len(labeled) - longs
        print(f"  Distribution: {longs} LONG, {shorts} SHORT")

        all_labeled.extend(labeled)

    if all_labeled:
        out_path = labeled_dir / "dataset.jsonl"
        save_labeled_dataset(all_labeled, out_path)
        print(f"\nDataset saved: {out_path} ({len(all_labeled)} samples)")
    else:
        print("No labeled samples generated.")


def cmd_run_backtest(args):
    candles_dir = str(Path(__file__).parent / "backtest" / "data" / "candles")

    result = asyncio.run(
        run_backtest(
            dataset_path=args.dataset,
            candles_dir=candles_dir,
            tag=args.tag,
            max_samples=args.max_samples,
            provider=args.provider,
            model=args.model,
            verbose=args.verbose,
        )
    )
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


def main():
    parser = argparse.ArgumentParser(description="Backtest Framework CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # fetch-candles
    p = sub.add_parser("fetch-candles", help="Download historical candles from Binance")
    p.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g. BTCUSDT,ETHUSDT)")
    p.add_argument("--interval", default="1h", help="Base candle interval (default: 1h)")
    p.add_argument("--months", type=int, default=6, help="Months of history (default: 6)")
    p.add_argument("--timeframes", default="1h,4h,1d", help="Comma-separated timeframes to fetch (default: 1h,4h,1d)")

    # prepare-dataset
    p = sub.add_parser("prepare-dataset", help="Calculate indicators and generate labeled dataset")
    p.add_argument("--symbols", required=True, help="Comma-separated symbols")
    p.add_argument("--interval", default="1h", help="Base candle interval (default: 1h)")
    p.add_argument("--timeframes", default="1h,4h,1d", help="Comma-separated timeframes to use (default: 1h,4h,1d)")

    # run-backtest
    p = sub.add_parser("run-backtest", help="Run backtest with LLM predictions")
    p.add_argument("--dataset", required=True, help="Path to labeled JSONL dataset")
    p.add_argument("--provider", help="LLM provider (groq, deepseek, mock, etc.)")
    p.add_argument("--model", help="LLM model name")
    p.add_argument("--tag", default="backtest", help="Tag for this run")
    p.add_argument("--max-samples", type=int, help="Limit number of samples")
    p.add_argument("--verbose", "-v", action="store_true", help="Show prompts, responses, and trade results per sample")

    # compare
    p = sub.add_parser("compare", help="Compare two backtest runs")
    p.add_argument("--baseline", required=True, help="Path to baseline results JSON")
    p.add_argument("--candidate", required=True, help="Path to candidate results JSON")

    # export-training-data
    p = sub.add_parser("export-training-data", help="Export labeled dataset as chat-format JSONL for fine-tuning")
    p.add_argument("--dataset", required=True, help="Path to labeled JSONL dataset")
    p.add_argument("--output", required=True, help="Output directory for train/val/test JSONL files")
    p.add_argument("--mode", default="FUTURES", choices=["SPOT", "FUTURES"], help="Trading mode (default: FUTURES)")

    args = parser.parse_args()

    commands = {
        "fetch-candles": cmd_fetch_candles,
        "prepare-dataset": cmd_prepare_dataset,
        "run-backtest": cmd_run_backtest,
        "compare": cmd_compare,
        "export-training-data": cmd_export_training_data,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
