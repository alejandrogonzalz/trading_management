"""Class-based data pipeline: fetch candles → calculate indicators → label → save dataset."""

import asyncio
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from backtest.config import DEFAULT_MONTHS, DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES
from backtest.ingestion.fetcher import DATA_DIR as CANDLES_DIR, fetch_multi_tf_candles, load_candles, save_candles
from backtest.ingestion.indicators import calculate_indicators_batch, calculate_multi_tf_indicators
from backtest.ingestion.labeler import generate_labeled_dataset, save_labeled_dataset

LABELED_DIR = Path(__file__).parent / "data" / "labeled"

# Minimal lookback per TF so we maximise usable output points.
# Chosen to satisfy MACD(26) warm-up on short TFs and keep weekly TF usable.
_HTF_LOOKBACK: Dict[str, int] = {
    "5m": 40,
    "15m": 40,
    "30m": 40,
    "1h": 200,   # base TF — full lookback for EMA200
    "4h": 40,
    "1d": 40,
    "1w": 30,
}
_DEFAULT_HTF_LOOKBACK = 40


class DataPipeline:
    """Orchestrates: fetch → indicators → label → save.

    Usage::

        pipe = DataPipeline(symbols=["BTCUSDT"], timeframes=["15m","1h","4h","1d"])
        asyncio.run(pipe.fetch())   # download candles
        pipe.build_dataset()        # indicators + labeling
        # dataset at data/labeled/dataset.jsonl
    """

    def __init__(
        self,
        symbols: List[str] = DEFAULT_SYMBOLS,
        timeframes: List[str] = DEFAULT_TIMEFRAMES,
        months: int = DEFAULT_MONTHS,
        base_tf: str = "1h",
        lookahead: int = 24,
        output_path: Optional[Path] = None,
        candles_dir: Optional[Path] = None,
    ):
        self.symbols = symbols
        self.timeframes = timeframes
        self.months = months
        self.base_tf = base_tf
        self.lookahead = lookahead
        self.output_path = output_path or (LABELED_DIR / "dataset.jsonl")
        self.candles_dir = candles_dir or CANDLES_DIR

        if base_tf not in timeframes:
            raise ValueError(f"base_tf='{base_tf}' must be in timeframes={timeframes}")

    # ------------------------------------------------------------------
    # Step 1 — fetch
    # ------------------------------------------------------------------

    def _date_range(self) -> tuple[str, str]:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=self.months * 30)
        return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")

    async def fetch(self, skip_existing: bool = False) -> None:
        """Download candle files for all symbols × timeframes concurrently."""
        start_str, end_str = self._date_range()

        async def _fetch_symbol(sym: str) -> None:
            tfs_to_fetch = self.timeframes
            if skip_existing:
                tfs_to_fetch = [
                    tf for tf in self.timeframes
                    if not (self.candles_dir / f"{sym}_{tf}.json").exists()
                ]
            if not tfs_to_fetch:
                print(f"  {sym}: all candle files already exist, skipping")
                return

            print(f"Fetching {sym}: {tfs_to_fetch} — {start_str} to {end_str}")
            candles_by_tf = await fetch_multi_tf_candles(sym, tfs_to_fetch, start_str, end_str)
            for tf, candles in candles_by_tf.items():
                path = save_candles(candles, sym, tf)
                print(f"  Saved {len(candles)} {tf} candles → {path.name}")

        for sym in self.symbols:
            await _fetch_symbol(sym)

    # ------------------------------------------------------------------
    # Step 2 — indicators + labeling
    # ------------------------------------------------------------------

    def _load_candles_for_symbol(self, sym: str) -> Dict[str, List[Dict[str, Any]]]:
        available: Dict[str, List[Dict[str, Any]]] = {}
        for tf in self.timeframes:
            candles = load_candles(sym, tf, self.candles_dir)
            if candles:
                available[tf] = candles
        return available

    def _indicators_for_symbol(self, sym: str, candles_by_tf: Dict[str, List]) -> List[Dict[str, Any]]:
        lookback = _HTF_LOOKBACK.get(self.base_tf, 200)

        # Build per-TF lookback map for calculate_multi_tf_indicators
        # Patch the module-level constant temporarily via kwargs isn't possible,
        # so we call calculate_indicators_batch directly for each TF and merge.
        # calculate_multi_tf_indicators handles arbitrary TFs already.
        try:
            return calculate_multi_tf_indicators(candles_by_tf, base_tf=self.base_tf, lookback=lookback)
        except ValueError as exc:
            print(f"  Skipping {sym}: {exc}")
            return []

    def build_dataset(self) -> List[Dict[str, Any]]:
        """Calculate indicators and label all symbols. Returns all labeled samples."""
        all_labeled: List[Dict[str, Any]] = []

        for sym in self.symbols:
            candles_by_tf = self._load_candles_for_symbol(sym)

            if self.base_tf not in candles_by_tf:
                print(f"  {sym}: no {self.base_tf} candles, run fetch() first")
                continue

            mode = "multi-TF" if len(candles_by_tf) > 1 else "single-TF"
            print(f"Indicators {sym} ({mode}): {list(candles_by_tf.keys())}")

            indicators = self._indicators_for_symbol(sym, candles_by_tf)
            if not indicators:
                continue
            print(f"  {len(indicators)} indicator points")

            labeled = generate_labeled_dataset(
                candles_by_tf[self.base_tf], indicators, sym, lookahead=self.lookahead
            )
            longs = sum(1 for s in labeled if s["label"]["bias"] == "LONG")
            print(f"  {len(labeled)} labeled samples  ({longs} LONG / {len(labeled) - longs} SHORT)")

            all_labeled.extend(labeled)

        if all_labeled:
            save_labeled_dataset(all_labeled, self.output_path)
            print(f"\nDataset saved → {self.output_path}  ({len(all_labeled)} samples)")
        else:
            print("No labeled samples generated.")

        return all_labeled

    def run(self, skip_existing_candles: bool = False) -> List[Dict[str, Any]]:
        """Fetch + build dataset in one call."""
        asyncio.run(self.fetch(skip_existing=skip_existing_candles))
        return self.build_dataset()
