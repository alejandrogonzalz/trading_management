# Backtest Trade Simulation — How It Works

## Overview

The backtest evaluates model predictions by simulating trades on real historical candle data. It does NOT simulate with dollar amounts — everything is measured in **percentage returns per trade** (flat bet, no compounding).

## Two Independent Metrics

| Metric | Question | Uses TP/SL? | Affected by circularity? |
|--------|----------|-------------|--------------------------|
| **Direction accuracy** | Did you correctly predict LONG vs SHORT? | No | No |
| **Win rate / Profit factor** | Did the trade reach TP before SL? | Yes | Depends on simulation mode |

These are computed separately. A model can have 88% accuracy but only 50% win rate because predicting the right direction doesn't guarantee the price reaches your take-profit level before hitting stop-loss.

## How Direction Accuracy Works

```
For each test sample:
  1. Label says: "LONG" (price went up more than it went down in next 24h)
  2. Model predicts: "LONG" or "SHORT"
  3. If model == label → correct

Accuracy = correct / total
```

TP/SL play zero role here. It's a pure classification metric.

## How Trade Simulation Works

Two simulation modes exist in `backtest/evaluation/simulate.py`:

### Mode 1: Hindsight TP/SL (`simulate_trade()`) — CIRCULAR, for ranking only

Uses the TP/SL the model predicted (which it learned from labels that were calculated from future prices):

```python
# For a LONG trade:
for each of the next 24 candles (1h each):
    if candle.low <= SL:  → LOSS, pnl = (SL - entry) / entry * 100 - fee
    if candle.high >= TP:  → WIN,  pnl = (TP - entry) / entry * 100 - fee
if neither hit:            → TIMEOUT, pnl = (last_close - entry) / entry * 100 - fee
```

**Why it's circular**: The labeler computed `TP = entry * (1 + max_future_up * 0.7)` — it looked at how high the price ACTUALLY went and placed TP at 70% of that maximum. The model learned to replicate this value. The simulator then asks "did price reach the model's TP?" — almost always YES, because the TP was designed knowing the price went higher. This produces PF=12.92, which is an artifact.

### Mode 2: ATR-based TP/SL (`simulate_trade_atr()`) — LEGITIMATE

Ignores the model's predicted TP/SL entirely. Uses only the current ATR (backward-looking, available at decision time):

```python
# For a LONG trade:
TP = entry + ATR * 1.5   # fixed 1.5:1 reward:risk ratio
SL = entry - ATR * 1.0

# Same candle loop as above, but with these independent levels
```

**Why it's legitimate**: ATR is computed from the past 14 candles only. The exit levels are a pure function of current volatility — no future information.

### Mode comparison on `qlora_no_drawdown` (84.13% accuracy):

| Simulation | Win Rate | Profit Factor | Avg Win | Avg Loss |
|-----------|----------|---------------|---------|----------|
| Hindsight TP/SL | 50.55% | 4.13 | +2.63% | -0.91% |
| **ATR TP/SL** | **66.31%** | **2.09** | **+1.22%** | **-1.15%** |

The ATR numbers are what you'd realistically see in production.

## What the Simulation Does NOT Include

- **Dollar amounts**: no position sizing, always "1 unit" per trade
- **Compounding**: each trade is independent, profits don't grow the account
- **Slippage**: fills assumed at exact TP/SL price
- **Market impact**: no modeling of how your order moves the market
- **Partial fills**: always fully filled
- **Concurrent trades**: simulated sequentially, no capital constraint

## Profit Factor Explained

```
PF = total_gross_profit / total_gross_loss

Where:
  total_gross_profit = sum of pnl_pct for all trades with pnl > 0
  total_gross_loss = abs(sum of pnl_pct for all trades with pnl < 0)
```

Interpretation:
- PF = 1.0 → breakeven
- PF = 1.5-2.5 → good system (professional range, per Pardo 2008)
- PF = 3.0+ → excellent or suspicious
- PF > 5.0 → almost certainly overfit or has methodology issues
- PF = 12.92 → artifact (our hindsight simulation)

## The Drawdown Filter vs TP/SL Circularity

These are two separate issues:

| Issue | What it does | What it affects | Fix |
|-------|-------------|-----------------|-----|
| **Drawdown filter** | Discards samples where SL hit before TP | Dataset composition (easier/harder) | `--no-drawdown-filter` |
| **TP/SL circularity** | Labels contain exit prices derived from future data | Financial metrics (PF, WR) | `--use-atr-tp-sl` or `simulate_trade_atr()` |

The drawdown filter controls **which samples enter the dataset**. TP/SL circularity controls **how exit prices are calculated**. They're orthogonal — you can (and should) address both independently.

With `--no-drawdown-filter`: noisy samples are KEPT (SL might trigger before TP), but the TP/SL values are STILL derived from future prices. Accuracy drops (84% vs 88%) because the task is harder. Financial metrics are still circular unless you use ATR simulation.

## Timeframe Structure

- **Base candle**: 1 hour (each sample = one decision point)
- **Indicators**: computed on 1h + 4h + 1d (multi-timeframe alignment)
- **Max hold**: 24 candles = 24 hours per trade
- **Test period**: 81 days (Feb 15 — May 7, 2026)
- **Setups per day**: ~104 pass all filters across 12 symbols
- **Fee**: 0.1% per side (0.2% round-trip, Binance taker rate)

## Which Results to Present

### For the thesis (recommendation):

| What to report | Source | Why |
|---------------|--------|-----|
| **Direction accuracy: 84.13%** | `qlora_no_drawdown` | Hardest dataset (no survivorship filter), most honest |
| **Direction accuracy: 88.03%** | `qlora_cloud` | Main result on filtered (operational) dataset |
| **Financial metrics: ATR only** | `metrics_atr` from any run | Only legitimate simulation |
| **PF = 2.09, WR = 66.3%** | `qlora_no_drawdown.metrics_atr` | Realistic on hard data |
| **PF = 4.63, WR = 81.7%** | `qlora_anon_symbol.metrics_atr` | Realistic on filtered data |

### Do NOT present as evidence of viability:

- PF = 12.92 (hindsight, circular)
- PF = 4.13 (hindsight on no-filter data, still circular)
- Any equity curve from hindsight simulation (compounds to 1e60x)

## Recommended Default: `simulate_trade_atr()` always

The ATR simulation should be the **primary** financial metric in any report. The hindsight simulation should only be mentioned to explain why it's wrong (methodology discussion). The evaluator already runs both automatically — just cite `metrics_atr`, not `test_metrics`.
