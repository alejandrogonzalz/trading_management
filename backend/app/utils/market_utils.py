import pandas as pd
import numpy as np
import talib
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from app.services import binance_service
from app.services import market_service

def compute_metrics_for_symbol(c: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Worker to compute opportunity metrics for a single symbol."""
    try:
        symbol = c['symbol']
        candles = market_service.get_candles(symbol, "5m", limit=200)
        if not candles or len(candles) < 100:
            return None

        df = pd.DataFrame(candles)
        for col in ['high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)

        # Base Metrics
        atr = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        atr_norm_base = atr / df['close'].iloc[-1]
        rsi = talib.RSI(df['close'], timeperiod=14).iloc[-1]
        momentum_score = abs(rsi - 50)
        adx = talib.ADX(df['high'], df['low'], df['close'], timeperiod=14).iloc[-1]
        price_change = abs(df['close'].iloc[-1] - df['close'].iloc[-5]) / df['close'].iloc[-5]

        return {
            "symbol": symbol,
            "volume_24h": c['volume_24h'],
            "atr_rel": atr_norm_base,
            "momentum": momentum_score,
            "adx": adx,
            "price_move": price_change
        }
    except Exception:
        return None

def get_top_opportunity_pairs(limit: int = 20) -> List[str]:
    """Ranks USDC and USDT markets in parallel."""
    print("Fetching market-wide tickers and exchange info...")
    tickers = binance_service.binance_client.get_ticker()
    exchange_info = binance_service.binance_client.get_exchange_info()
    
    # Map status for quick lookup
    status_map = {s['symbol']: s['status'] for s in exchange_info['symbols']}
    
    candidates = []
    for t in tickers:
        symbol = t['symbol']
        volume = float(t['quoteVolume'])
        status = status_map.get(symbol, 'BREAK')
        
        if (symbol.endswith('USDC') or symbol.endswith('USDT')) and volume > 1_000_000 and status == 'TRADING':
            candidates.append({"symbol": symbol, "volume_24h": volume})

    if not candidates:
        return ["BTCUSDC", "ETHUSDC", "SOLUSDC", "BTCUSDT", "ETHUSDT", "SOLUSDT"]

    print(f"Analyzing {len(candidates)} candidates in parallel...")
    
    # Process all 50+ candidates at once
    with ThreadPoolExecutor(max_workers=20) as executor:
        results = list(executor.map(compute_metrics_for_symbol, candidates))
    
    market_data = [r for r in results if r is not None]

    if not market_data:
        return [c['symbol'] for c in candidates[:limit]]

    df_ranking = pd.DataFrame(market_data)
    for col in ["volume_24h", "atr_rel", "momentum", "adx", "price_move"]:
        max_val = df_ranking[col].max()
        df_ranking[f"{col}_norm"] = df_ranking[col] / max_val if max_val > 0 else 0

    df_ranking['opp_score'] = (
        0.30 * df_ranking['volume_24h_norm'] +
        0.25 * df_ranking['atr_rel_norm'] +
        0.20 * df_ranking['momentum_norm'] +
        0.15 * df_ranking['adx_norm'] +
        0.10 * df_ranking['price_move_norm']
    )

    top_20 = df_ranking.sort_values(by="opp_score", ascending=False)['symbol'].head(limit).tolist()
    print(f"Opportunity Discovery complete.")
    return top_20
