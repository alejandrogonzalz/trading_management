import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.services import indicator_service, market_service, scoring_service

# In-memory store for the latest scan results
latest_scan_results: dict[str, Any] = {}


def calculate_multi_tf_heatmap(symbol: str) -> dict[str, str]:
    """
    Calculates EMA trend across all timeframes.
    Note: market_service.get_multi_timeframe_candles is already parallelized.
    """
    multi_tf_candles = market_service.get_multi_timeframe_candles(symbol)
    heatmap_results = {}
    for tf, candles in multi_tf_candles.items():
        if not candles:
            heatmap_results[tf] = "NEUTRAL"
            continue
        indicators = indicator_service.calculate_indicators(candles)
        heatmap_results[tf] = indicators.get("heatmap", "NEUTRAL")
    return heatmap_results


def process_single_pair(pair: str, base_tf: str) -> dict[str, Any] | None:
    """
    Worker function to process one pair completely.
    """
    try:
        # 1. Fetch data for base timeframe
        candles = market_service.get_candles(pair, base_tf)
        if not candles:
            return None

        indicators = indicator_service.calculate_indicators(candles)
        if indicators.get("error"):
            return None

        # 2. Confluence Score
        score_data = scoring_service.calculate_score(indicators)

        # 3. Parallel Multi-TF Heatmap
        heatmap_multi = calculate_multi_tf_heatmap(pair)

        return {
            "pair": pair,
            "price": indicators.get("price"),
            "timeframe": base_tf,
            "heatmap": indicators.get("heatmap"),
            "heatmap_multi": heatmap_multi,
            "structure": indicators.get("structure"),
            "rsi": indicators.get("rsi"),
            "macd_hist": indicators.get("macd_hist"),
            "adx": indicators.get("adx"),
            "volume_ratio": indicators.get("volume_ratio"),
            "atr_ratio": indicators.get("atr_ratio"),
            "bb_pos": indicators.get("bb_pos"),
            "score": score_data.get("score"),
        }
    except Exception as e:
        print(f"Error processing {pair}: {e}")
        return None


def run_scan(pairs: list[str], base_tf: str = "1h") -> list[dict[str, Any]]:
    """
    Runs a scan for all pairs IN PARALLEL.
    """
    global latest_scan_results
    scan_start_time = datetime.datetime.now().isoformat()

    print(f"Starting Turbo-Scan for {len(pairs)} pairs on {base_tf}...")

    results = []
    # Use 20 workers to process all pairs at once
    with ThreadPoolExecutor(max_workers=20) as executor:
        # map process_single_pair across all pairs
        future_results = list(executor.map(lambda p: process_single_pair(p, base_tf), pairs))

        # Filter out None results (failed pairs)
        results = [r for r in future_results if r is not None]

    latest_scan_results = {"timestamp": scan_start_time, "base_timeframe": base_tf, "results": results}
    print(
        f"Turbo-Scan completed in {(datetime.datetime.now() - datetime.datetime.fromisoformat(scan_start_time)).total_seconds():.2f} seconds."
    )
    return results


def get_latest_scan() -> dict[str, Any]:
    return latest_scan_results
