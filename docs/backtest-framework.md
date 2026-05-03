# Backtest & Comparison Framework

**Date**: April 30, 2026
**Project**: Trading Management System — Maestría en IA Aplicada, Tec de Monterrey
**Author**: Alejandro González Almazán

---

## Overview

A framework to evaluate LLM-generated trade setups against historical market data. Compares a zero-shot baseline (Qwen 2.5 via Groq/DeepSeek) against a fine-tuned model (Together AI) using classification metrics and simulated P&L.

**Related docs**:
- [Fine-Tuning Strategy](./fine-tuning-strategy.md) — training data pipeline and model training
- [Action Plan](./action-plan.md) — implementation timeline and dependencies

---

## 1. Provider Changes

### OpenAICompatibleProvider

The existing `llm_factory.py` has 5 providers. DeepSeek, Groq, Together, and Fireworks are all OpenAI-compatible — one unified provider replaces the current `OpenAIProvider`:

```python
class OpenAICompatibleProvider(LLMProvider):
    """Works with OpenAI, DeepSeek, Groq, Together, Fireworks."""
    def __init__(self, model_name: str, temperature: float,
                 base_url: Optional[str] = None, api_key: Optional[str] = None):
        from langchain_openai import ChatOpenAI
        kwargs = {"model": model_name, "temperature": temperature}
        if base_url:
            kwargs["base_url"] = base_url
        if api_key:
            kwargs["api_key"] = api_key
        self.client = ChatOpenAI(**kwargs)
```

### Factory Update

The factory auto-resolves base URLs by provider name:

| LLM_PROVIDER | LLM_MODEL | LLM_BASE_URL (auto) |
|---|---|---|
| `deepseek` | `deepseek-chat` | `https://api.deepseek.com/v1` |
| `groq` | `llama-3.3-70b-versatile` | `https://api.groq.com/openai/v1` |
| `together` | `Qwen/Qwen2.5-7B-Instruct` or fine-tuned ID | `https://api.together.xyz/v1` |
| `fireworks` | any supported model | `https://api.fireworks.ai/inference/v1` |
| `openai` | `gpt-4o-mini` | (default OpenAI) |

### Recommended Models for Backtest

- **Baseline (cheap, fast)**: Groq `llama-3.3-70b-versatile` — free tier, ~6000 req/day
- **Baseline (quality)**: DeepSeek `deepseek-chat` — $0.14/M input tokens
- **Fine-tuned**: Together `alexglz/Qwen2.5-7B-Instruct-trading-v1`

### Files Changed

| File | Change |
|---|---|
| `langgraph/agent/llm_factory.py` | Replace `OpenAIProvider` with `OpenAICompatibleProvider`, update factory |
| `.env.example` | Add `LLM_BASE_URL`, `LLM_API_KEY` examples |

---

## 2. Data Pipeline

```
1. FETCH CANDLES        Binance public API → raw OHLCV (no API key needed)
2. CALCULATE INDICATORS Sliding window over candles using indicator_service
3. GENERATE LABELS      Hindsight labeling — look ahead N candles for ground truth
4. FEED TO LLM          Send indicators to LangGraph or direct LLM call
5. SIMULATE TRADES      Compare predictions vs actual price movement
```

### Fetching Candles

Binance public klines endpoint — no API key required:

```
GET https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1h&limit=1000&startTime=...
```

Up to 1000 candles per request. Paginate by advancing `startTime`. Rate limit: 1200 req/min.

### Labeling Logic

For each candle at time T, look ahead to determine the correct trade direction. Uses ATR-based dynamic thresholds (not fixed percentages) and checks drawdown-before-profit (path dependency).

Key design decisions:
- **Dynamic threshold**: 1.5× ATR minimum move (adapts to each pair's volatility)
- **Directional clarity**: Up must be > 1.5× Down (or vice versa) to avoid ambiguous labels
- **Whipsaw filter**: Skip if direction reverses within 4 candles
- **No NEUTRAL class**: Ambiguous examples are discarded, not labeled — see [Fine-Tuning Strategy](./fine-tuning-strategy.md#the-neutral-class-decision) for rationale

```python
def label_candle(candles, index, lookahead=24, atr_multiplier=1.5):
    entry = candles[index]["close"]
    future = candles[index + 1 : index + 1 + lookahead]
    atr = indicators[index]["atr_raw"]
    threshold_pct = (atr * atr_multiplier) / entry

    max_up = (max(c["high"] for c in future) - entry) / entry
    max_down = (entry - min(c["low"] for c in future)) / entry

    if max_up > threshold_pct and max_up > max_down * 1.5:
        return "LONG", calculate_tp_sl(...)
    elif max_down > threshold_pct and max_down > max_up * 1.5:
        return "SHORT", calculate_tp_sl(...)
    else:
        return None  # Discard — ambiguous
```

### Quality Filters

| Filter | Threshold | Rationale |
|---|---|---|
| Minimum move | 1.5× ATR | Below this is noise |
| Directional clarity | Up > 1.5× Down (or vice versa) | Ambiguous moves teach nothing |
| Volume ratio | > 0.5 | Low volume = unreliable |
| ADX | > 15 | Below 15 = random walk |
| Risk:Reward | > 1:1 | Not worth trading even if correct |
| Whipsaw | No reversal > threshold in first 4 candles | False signals |

---

## 3. Backtest Simulation

### Trade Simulation

```python
def simulate_trade(prediction, future_candles, max_hold=24):
    entry, tp, sl, bias = prediction["entry"], prediction["tp"], prediction["sl"], prediction["bias"]

    for i, candle in enumerate(future_candles[:max_hold]):
        if bias == "LONG":
            if candle["low"] <= sl:
                return {"outcome": "LOSS", "pnl_pct": (sl - entry) / entry * 100}
            if candle["high"] >= tp:
                return {"outcome": "WIN", "pnl_pct": (tp - entry) / entry * 100}
        elif bias == "SHORT":
            if candle["high"] >= sl:
                return {"outcome": "LOSS", "pnl_pct": (entry - sl) / entry * 100}
            if candle["low"] <= tp:
                return {"outcome": "WIN", "pnl_pct": (entry - tp) / entry * 100}

    # Timeout — close at last candle
    last = future_candles[min(max_hold - 1, len(future_candles) - 1)]["close"]
    pnl = ((last - entry) / entry * 100) if bias == "LONG" else ((entry - last) / entry * 100)
    return {"outcome": "TIMEOUT", "pnl_pct": pnl}
```

### Metrics

| Metric | Formula | What It Measures |
|---|---|---|
| Direction Accuracy | correct / total | Did the model predict LONG/SHORT correctly? |
| Precision (per class) | TP / (TP + FP) | When it says LONG, how often is it right? |
| Recall (per class) | TP / (TP + FN) | Of all actual LONGs, how many did it catch? |
| F1 Score | 2 × P × R / (P + R) | Harmonic mean of precision and recall |
| Win Rate | wins / total_trades | % of trades that hit TP before SL |
| Profit Factor | gross_profit / gross_loss | >1 = profitable system |
| Sharpe Ratio | mean(returns) / std(returns) × √N | Risk-adjusted return |
| Max Drawdown | max peak-to-trough in equity curve | Worst losing streak |
| Confidence Calibration | bin by confidence, compare to actual accuracy | Is confidence=80 really 80% accurate? |

---

## 4. File Structure

```
langgraph/
├── agent/
│   └── llm_factory.py              ← MODIFY: add OpenAICompatibleProvider
├── backtest/
│   ├── __init__.py
│   ├── fetch_candles.py            ← Download historical OHLCV from Binance
│   ├── calculate_indicators.py     ← Batch indicator calculation
│   ├── label_data.py               ← Hindsight labeling (LONG/SHORT)
│   ├── run_backtest.py             ← Feed data to LLM, simulate trades
│   ├── compare.py                  ← Side-by-side comparison of two runs
│   ├── metrics.py                  ← All metric calculations
│   └── report.py                   ← Terminal table + matplotlib charts
├── backtest/data/                  ← gitignored, generated at runtime
│   ├── candles/                    ← Raw OHLCV CSVs per symbol
│   ├── labeled/                    ← Labeled datasets (JSONL)
│   └── results/                    ← Backtest output JSONs per model run
└── cli.py                          ← CLI entry point
```

---

## 5. CLI Interface

```bash
# Download historical candles
python -m langgraph.cli fetch-candles --symbols BTCUSDT,ETHUSDT --interval 1h --months 6

# Calculate indicators and generate labeled dataset
python -m langgraph.cli prepare-dataset --symbols BTCUSDT,ETHUSDT

# Run backtest with a specific provider
python -m langgraph.cli run-backtest \
    --dataset backtest/data/labeled/dataset.jsonl \
    --provider groq --model llama-3.3-70b-versatile --tag baseline-groq

# Compare two backtest runs
python -m langgraph.cli compare \
    --baseline results/baseline-groq.json \
    --candidate results/finetuned-v1.json
```

---

## 6. Comparison Report Format

```
╔══════════════════════════════════════════════════════════════╗
║          BACKTEST COMPARISON — BTCUSDT 1h (6 months)        ║
╠══════════════════════════════════════════════════════════════╣
║ Metric                │ Baseline (Groq)  │ Fine-tuned (v1)  ║
╠═══════════════════════╪══════════════════╪══════════════════╣
║ Direction Accuracy    │ 52.3%            │ 67.8%            ║
║ LONG Precision        │ 55.1%            │ 71.2%            ║
║ SHORT Precision       │ 48.7%            │ 63.4%            ║
║ F1 Score              │ 0.51             │ 0.68             ║
║ Win Rate              │ 45.2%            │ 58.9%            ║
║ Profit Factor         │ 0.87             │ 1.42             ║
║ Sharpe Ratio          │ -0.12            │ 0.85             ║
║ Max Drawdown          │ -18.3%           │ -9.7%            ║
║ Avg Latency           │ 320ms            │ 450ms            ║
╚═══════════════════════╧══════════════════╧══════════════════╝
```

Plus matplotlib charts: equity curve, confusion matrix, confidence calibration diagram.

---

## 7. Implementation Phases

| Phase | What | Depends On | Can Start Now? |
|---|---|---|---|
| **1. Provider changes** | `OpenAICompatibleProvider` in `llm_factory.py` | Nothing | ✅ Yes |
| **2. Data pipeline** | `fetch_candles.py`, `calculate_indicators.py`, `label_data.py` | Nothing | ✅ Yes |
| **3. Backtest runner** | `run_backtest.py`, `metrics.py`, `report.py` | Phase 2 | ✅ After Phase 2 |
| **4. Comparison framework** | `compare.py`, CLI `compare` command | Phase 3 | ✅ After Phase 3 |
| **5. Fine-tuned evaluation** | Run backtest with fine-tuned model | Fine-tuning done | ❌ After training |

Phases 1-4 can be built and tested with baseline models (Groq free tier, DeepSeek). Phase 5 only needs the fine-tuned model.

### Dependencies to Add

```
# langgraph/requirements.txt — add:
matplotlib     # charts
pandas         # data manipulation
tabulate       # terminal tables
```

No new LangChain dependencies — `langchain-openai` already handles all OpenAI-compatible APIs.
