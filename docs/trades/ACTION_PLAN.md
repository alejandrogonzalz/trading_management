# Action Plan — Trade Simulation & Deliverables

## Answers to Your Questions

## How to Improve Trade Simulation (Future Work)

Improvement priority chain: **Variable ATR multipliers > slippage model > position sizing > walk-forward > paper trading**. None of these are necessary for the defense; all go in "future work."

### Priority 1: Variable TP/SL multipliers (medium effort)

Instead of fixed 1.5×ATR for TP and 1.0×ATR for SL, train the model to output the **optimal multiplier** per trade: `{"bias": "LONG", "tp_mult": 1.8, "sl_mult": 0.7, "reasoning": "..."}`. How to label: for each historical sample, brute-force test combinations (tp_mult ∈ [1.0, 1.5, 2.0, 2.5], sl_mult ∈ [0.5, 0.75, 1.0, 1.5]) and pick the one that maximized expectancy over the 24h window. This is still somewhat circular (uses future to pick the best combo) but much less so than the current approach (which uses the exact future price peak).

### Priority 2: Transaction costs + realistic fills (low effort)

Already partially done (fee=0.1%/side). Add: slippage model (5-20 bps depending on symbol liquidity), fill assumption where SL fills at SL ± slippage (not exact), and concurrency cap of max 3 active trades at once (capital constraint). Implementation: modify `simulate_trade_atr()` to accept `slippage_bps` param.

### Priority 3: Position sizing (medium effort)

Currently every trade is "1 unit". A realistic system would vary size by confidence score from the model (higher confidence → larger position), current volatility (higher ATR → smaller position to maintain fixed dollar risk), and Kelly criterion: f* = (WR × avg_win/avg_loss - (1-WR)) / (avg_win/avg_loss). For the thesis this is overkill. For production it's essential.

### Priority 4: Walk-forward validation (high effort, high value)

Train on months 1-12, test on month 13. Retrain on 1-13, test on 14. Repeat. This tests whether the model generalizes across regime changes. Cost: ~$5/window × 4 windows = $20. Time: ~14h total.

### Priority 5: Paper trading (production only)

Run the model live with real-time data but fake orders for 4+ weeks. Monitor: rolling accuracy, regime detection, latency, edge decay.

The story for the defense is: "direction accuracy is real and significant (+29pp over zero-shot); financial metrics are preliminary but positive under legitimate ATR simulation (PF=2.09 on hard data); production deployment requires additional validation (walk-forward, paper trading) before committing capital."
