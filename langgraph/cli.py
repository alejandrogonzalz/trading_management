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

from backtest.fetch_candles import fetch_candles, save_candles  # noqa: E402
from backtest.calculate_indicators import calculate_indicators_batch  # noqa: E402
from backtest.label_data import generate_labeled_dataset, save_labeled_dataset  # noqa: E402
from backtest.run_backtest import run_backtest  # noqa: E402
from backtest.compare import compare  # noqa: E402
from backtest.report import print_report, print_comparison  # noqa: E402


def cmd_fetch_candles(args):
    symbols = [s.strip() for s in args.symbols.split(",")]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=args.months * 30)
    start_str = start.strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")

    async def _fetch():
        for sym in symbols:
            print(f"Fetching {sym} {args.interval} from {start_str} to {end_str}...")
            candles = await fetch_candles(sym, args.interval, start_str, end_str)
            path = save_candles(candles, sym, args.interval)
            print(f"  Saved {len(candles)} candles to {path}")

    asyncio.run(_fetch())


def cmd_prepare_dataset(args):
    symbols = [s.strip() for s in args.symbols.split(",")]
    candles_dir = Path(__file__).parent / "backtest" / "data" / "candles"
    labeled_dir = Path(__file__).parent / "backtest" / "data" / "labeled"

    all_labeled = []
    for sym in symbols:
        candle_file = candles_dir / f"{sym}_{args.interval}.json"
        if not candle_file.exists():
            print(f"No candle data for {sym}_{args.interval}. Run fetch-candles first.")
            continue

        with open(candle_file) as f:
            candles = json.load(f)

        print(f"Calculating indicators for {sym} ({len(candles)} candles)...")
        indicators = calculate_indicators_batch(candles)
        print(f"  Got {len(indicators)} indicator points")

        print(f"Labeling {sym}...")
        labeled = generate_labeled_dataset(candles, indicators, sym)
        print(f"  Got {len(labeled)} labeled samples (LONG/SHORT only, ambiguous discarded)")

        # Count distribution
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
        )
    )
    print_report(result)


def cmd_compare(args):
    comparison = compare(args.baseline, args.candidate)
    print_comparison(comparison)


def main():
    parser = argparse.ArgumentParser(description="Backtest Framework CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    # fetch-candles
    p = sub.add_parser("fetch-candles", help="Download historical candles from Binance")
    p.add_argument("--symbols", required=True, help="Comma-separated symbols (e.g. BTCUSDT,ETHUSDT)")
    p.add_argument("--interval", default="1h", help="Candle interval (default: 1h)")
    p.add_argument("--months", type=int, default=6, help="Months of history (default: 6)")

    # prepare-dataset
    p = sub.add_parser("prepare-dataset", help="Calculate indicators and generate labeled dataset")
    p.add_argument("--symbols", required=True, help="Comma-separated symbols")
    p.add_argument("--interval", default="1h", help="Candle interval (default: 1h)")

    # run-backtest
    p = sub.add_parser("run-backtest", help="Run backtest with LLM predictions")
    p.add_argument("--dataset", required=True, help="Path to labeled JSONL dataset")
    p.add_argument("--provider", help="LLM provider (groq, deepseek, mock, etc.)")
    p.add_argument("--model", help="LLM model name")
    p.add_argument("--tag", default="backtest", help="Tag for this run")
    p.add_argument("--max-samples", type=int, help="Limit number of samples")

    # compare
    p = sub.add_parser("compare", help="Compare two backtest runs")
    p.add_argument("--baseline", required=True, help="Path to baseline results JSON")
    p.add_argument("--candidate", required=True, help="Path to candidate results JSON")

    args = parser.parse_args()

    commands = {
        "fetch-candles": cmd_fetch_candles,
        "prepare-dataset": cmd_prepare_dataset,
        "run-backtest": cmd_run_backtest,
        "compare": cmd_compare,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
