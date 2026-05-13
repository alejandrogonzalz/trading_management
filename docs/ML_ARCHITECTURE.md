# ML Architecture — Models, Backtesting & LangGraph Integration

## Table of Contents
1. [ML Models vs Scanner Trend Metric](#1-ml-models-vs-scanner-trend-metric)
2. [How the Backtest Framework Works](#2-how-the-backtest-framework-works)
3. [How ML Models Improve Backtesting](#3-how-ml-models-improve-backtesting)
4. [LangGraph Integration — Current & Planned](#4-langgraph-integration--current--planned)

---

## 1. ML Models vs Scanner Trend Metric

### What the Scanner Does

The production scanner (`backend/app/services/scanner_service.py`) runs a multithreaded analysis across 20+ pairs and 7 timeframes. Its trend signal is a **deterministic, rule-based heatmap**:

```
EMA alignment (20/50/200) + ADX strength → heatmap
  STRONG_BULLISH | BULLISH | NEUTRAL | BEARISH | STRONG_BEARISH
```

This heatmap describes the **current market state** — it does not predict what will happen next. It is one input among many in the Quant Score ranking:

```
Quant Score (0-10):
  30% Volume  |  25% ATR (Volatility)  |  20% RSI (Momentum)
  15% ADX (Trend Strength)  |  10% Recent Move
```

### What the ML Models Do

The ML models (`langgraph/backtest/ml_models.py`) were trained specifically to answer one question:

> **"Given current multi-timeframe indicators, will the price go UP or DOWN in the next 24 candles?"**

They consume the **same indicators** the scanner produces — RSI, ADX, ATR, heatmap, structure, MACD, Bollinger Bands, volume — but across 3 timeframes simultaneously (1h, 4h, 1d), and learned from 56,161 labeled historical examples with hindsight.

### Head-to-Head Comparison

| Dimension | Scanner Heatmap | LSTM Model |
|-----------|----------------|------------|
| Purpose | Describe current state | Predict 24-candle direction |
| Logic | Deterministic rules | Learned from 56K examples |
| Multi-timeframe | 7 TFs (scanner) | 3 TFs (1h, 4h, 1d) |
| Output | State label | LONG / SHORT + confidence |
| Measured accuracy | No baseline | **85.5% on 8,425 unseen samples** |
| Knows when to override | No | Yes (RSI extremes, TF misalignment, volatility spikes) |

**Key insight**: The heatmap is an *input feature* of the LSTM — the model learned that EMA alignment matters, but also learned when to disagree with it. For example, if the heatmap says BULLISH but RSI is at 78 (overbought) on all 3 timeframes, the LSTM learned to predict SHORT.

### Test Set Results (8,425 samples never seen during training)

```
Direction Accuracy : 85.5%    ← vs ~50% random baseline
Win Rate           : 81.9%    ← trades that hit TP before SL
Profit Factor      : 7.02     ← gross gains / gross losses
Sharpe Ratio       : 18.16    ← simulation only, no fees/slippage
Max Drawdown       : 45.21%
LONG  F1           : 0.85
SHORT F1           : 0.86
```

> **Note for thesis / live use**: Sharpe and Profit Factor are from simulated trades with ATR-based TP/SL and no slippage or fees. These numbers serve for model comparison, not live P&L estimation.

### Model Ranking (CV scores, 56K samples, temporal split)

| Model | CV Accuracy | Test Accuracy | Overfitting Gap |
|-------|-------------|---------------|-----------------|
| **LSTM** | 84.29% | **85.5%** | None (generalizes) |
| XGBoost | 76.79% | TBD (round 2) | ⚠️ 0.187 |
| Random Forest | 74.89% | TBD (round 2) | ⚠️ 0.203 |
| Zero-shot LLM | TBD | — | — |
| Fine-tuned LLM (QLoRA) | TBD | — | — |

---

## 2. How the Backtest Framework Works

The backtest pipeline lives entirely in `langgraph/backtest/` and is orchestrated via `langgraph/cli.py`.

### Full Pipeline

```mermaid
flowchart TD
    subgraph FETCH["① FETCH CANDLES"]
        F1["fetch_candles.py"]
        F2["Binance API  /api/v3/klines"]
        F3["12 symbols × 3 timeframes × 18 months\n36 OHLCV JSON files · ~21 MB"]
        F1 --> F2 --> F3
    end

    subgraph CALC["② CALCULATE INDICATORS"]
        C1["calculate_indicators.py"]
        C2["TA-Lib: EMA 20/50/200 · RSI · MACD\nADX · ATR · Bollinger Bands · Vol SMA"]
        C3["Derived: heatmap · structure\natr_ratio · volume_ratio · bb_pos\nAligns 4h and 1d snapshot to each 1h candle"]
        C1 --> C2 --> C3
    end

    subgraph LABEL["③ HINDSIGHT LABELING"]
        L1["label_data.py\nLook 24 candles ahead per entry"]
        L2["Quality filters\nvol_ratio ≥ 0.5 · ADX ≥ 15 · R:R ≥ 1.0 · no whipsaw in first 4 candles"]
        L3["LONG if max_up &gt; 1.5× max_down  ·  SHORT if inverse\n56,161 labeled samples  ·  49% LONG · 51% SHORT"]
        L1 --> L2 --> L3
    end

    subgraph PREDICT["④ PREDICT"]
        subgraph LLM_PATH["LLM Path"]
            P1["run_backtest.py\nOllama / Groq / DeepSeek"]
            P2["→ bias · entry · tp · sl\nvia JSON prompt"]
            P1 --> P2
        end
        subgraph ML_PATH["ML Path"]
            P3["run_ml_backtest.py\nXGBoost · Random Forest · LSTM"]
            P4["→ bias + ATR-based tp/sl\n(temporal split: train 70 / val 15 / test 15)"]
            P3 --> P4
        end
    end

    subgraph SIM["⑤ TRADE SIMULATION"]
        S1["simulate_trade()"]
        S2["Scan next 24 candles for TP or SL hit\nFirst hit wins"]
        S3["Outcome: WIN · LOSS · TIMEOUT\npnl_pct = (exit − entry) / entry × direction"]
        S1 --> S2 --> S3
    end

    subgraph MET["⑥ METRICS  —  metrics.py"]
        M1["Classification\naccuracy · precision · recall · F1 per class"]
        M2["Trading\nwin_rate · profit_factor · avg win/loss\nsharpe ratio · max drawdown"]
        M3["Calibration\nbinned confidence vs realized accuracy"]
    end

    subgraph REP["⑦ COMPARE & REPORT"]
        R1["compare.py + report.py"]
        R2["Side-by-side table: baseline vs candidate\ncli: python -m cli compare --baseline f1.json --candidate f2"]
        R1 --> R2
    end

    FETCH --> CALC --> LABEL --> PREDICT --> SIM --> MET --> REP
```

### Temporal Split (critical — never shuffle)

```mermaid
flowchart LR
    subgraph TIMELINE["56,161 samples · ordered chronologically ──────────────────▶"]
        direction LR
        subgraph TRAIN["Train  70%"]
            TR["39,312 samples\nFit model weights"]
        end
        subgraph VAL["Val  15%"]
            VL["8,424 samples\nEarly stopping · CV"]
        end
        subgraph TEST["Test  15%"]
            TS["8,425 samples\n⚠ NEVER touched until final eval"]
        end
        TRAIN -->|time →| VAL -->|time →| TEST
    end
```

Shuffling would cause data leakage — future candles would appear in the training set. All splits respect chronological order.

### CLI Commands

```powershell
# From langgraph/ with conda activate trading

# 1. Download candles
python -m cli fetch-candles --symbols BTCUSDT,ETHUSDT,... --interval 1h --months 18 --timeframes "1h,4h,1d"

# 2. Generate labeled dataset
python -m cli prepare-dataset --symbols BTCUSDT,ETHUSDT,... --interval 1h --timeframes "1h,4h,1d"

# 3a. LLM backtest
python -m cli run-backtest --dataset backtest/data/labeled/dataset.jsonl --provider ollama

# 3b. ML backtest (auto-loads best params from optimization JSON)
python -m cli train-ml --model lstm --dataset backtest/data/labeled/dataset.jsonl --serialize

# 4. Compare two results
python -m cli compare --baseline backtest/data/results/baseline.json --candidate backtest/data/results/ml-lstm.json
```

---

## 3. How ML Models Improve Backtesting

### Before ML Models (LLM-only backtest)

The original backtest called an LLM for every sample:
- **Slow**: 100ms-2s per prediction (Ollama latency × 56K samples = hours)
- **Non-deterministic**: same indicators → different response each run
- **Provider-dependent**: results change with model version

### With ML Models

| | LLM Backtest | ML Backtest |
|---|---|---|
| Speed | Hours (56K API calls) | Minutes (vectorized inference) |
| Deterministic | No | Yes (same model = same predictions) |
| Interpretable | Via reasoning field | Via feature importance |
| Accuracy | TBD | LSTM 85.5%, XGB 76.79%, RF 74.89% |
| Deployable | Requires Ollama running | Single `.pt` or `.pkl` file |

### The Real Contribution: Labeled Dataset

The backtest framework's most valuable output is **not the model** — it's the **56,161 labeled samples with hindsight ground truth**. This dataset:
- Defines objectively when a LONG or SHORT was correct (not an opinion)
- Filters low-quality setups (low volume, low ADX, whipsaw)
- Enables fair comparison of any model — LLM, ML, or rule-based
- Will be used for QLoRA fine-tuning of Qwen 2.5 7B

---

## 4. LangGraph Integration — Current & Planned

### Current Architecture (Production)

The LangGraph agent (`langgraph/agent/`) runs independently from the ML backtest system. It receives live indicator data from the backend and generates trade setups through 3 nodes:

```mermaid
flowchart TD
    REQ(["POST /analyze · port 2024\nmulti-TF indicators from backend"])

    subgraph GEN["Node 1 — generator_node"]
        G1["Qwen 2.5 14B via Ollama"]
        G2["Generates trade setup\nbias · entry · tp · sl · leverage · quality"]
        G3["reasoning field — natural language explanation"]
        G1 --> G2 --> G3
    end

    subgraph EVAL["Node 2 — evaluator_node  ·  deterministic, no LLM"]
        E1["Structural checks\ntp/entry/sl ordering · R:R calculation"]
        E2["Market checks\nRSI overbought/oversold · ATR volatility ratio\nLiquidation distance · Multi-TF heatmap alignment"]
        E3["Output\nconfidence · issues list · rr · safety_margin"]
        E1 --> E2 --> E3
    end

    subgraph OPT["Node 3 — optimizer_node  ·  conditional"]
        O1["Qwen 2.5 14B — Risk Manager persona\nOnly triggered when evaluator finds issues"]
        O2["Refined entry · tp · sl\n+ changes list explaining each adjustment"]
        O1 --> O2
    end

    DONE(["Final trade setup"])

    REQ --> GEN --> EVAL
    EVAL -->|"issues found"| OPT --> DONE
    EVAL -->|"no issues"| DONE
```

**Current limitation**: The generator_node relies entirely on LLM intuition. There is no objective signal from the trained LSTM to anchor or validate the generated bias.

### Planned Integration — ML Model as Signal Node

The trained LSTM can be integrated as a **pre-generator validation step** or a **parallel signal node**:

```mermaid
flowchart TD
    REQ(["POST /analyze\nmulti-TF indicators"])

    subgraph ML0["Node 0  NEW — ml_signal_node"]
        M1["Loads lstm_final.pt once at startup\nno GPU needed"]
        M2["CPU inference  ·  &lt; 1ms latency"]
        M3["Output added to TradeState\nml_bias: LONG/SHORT  ·  ml_confidence: 0-10"]
        M1 --> M2 --> M3
    end

    subgraph GEN["Node 1 — generator_node  ·  enhanced"]
        G1["Qwen 2.5 14B via Ollama"]
        G2["System prompt includes ML prior\n'LSTM (85.5% accuracy) predicts LONG · confidence 8/10\nFactor this in or explain if you disagree'"]
        G3["→ Anchored output: bias · entry · tp · sl"]
        G1 --> G2 --> G3
    end

    subgraph TAIL["Nodes 2 & 3 — unchanged"]
        T1["evaluator_node\nNow also flags LLM ↔ ML bias disagreement as an issue"]
        T2["optimizer_node\nConditional refinement if issues found"]
        T1 -->|"if issues"| T2
    end

    DONE(["Trade Setup\nML-grounded"])

    REQ --> ML0 --> GEN --> TAIL --> DONE
```

**Why this matters**:
- LSTM runs in <1ms on CPU — zero latency cost
- Provides a statistically grounded prior to the LLM
- When LLM and LSTM agree → higher confidence signal
- When they disagree → evaluator flags this as an issue for optimizer

### Integration with the Scanner (Backend)

The scanner already collects live multi-TF indicators. The connection would be:

```mermaid
flowchart LR
    subgraph BACKEND["Backend Scanner"]
        SC["scanner_service.py\n20+ pairs · 7 timeframes · multithreaded"]
        IND["Indicators per TF\nRSI · ADX · ATR · heatmap\nstructure · MACD · BB · volume"]
        SC --> IND
    end

    subgraph INFERENCE["LSTM Inference Layer"]
        LM["lstm_final.pt\nloaded once at startup"]
        CPU["CPU · &lt; 1ms\nno GPU required"]
        SIG["ml_bias: LONG / SHORT\nml_confidence: 0–10"]
        LM --> CPU --> SIG
    end

    subgraph AGENT["LangGraph Agent  ·  port 2024"]
        N0["ml_signal_node\ninjects ML prior into state"]
        N1["generator_node\nLLM with ML-anchored prompt"]
        N2["evaluator + optimizer\nflags LLM ↔ ML disagreement"]
        N0 --> N1 --> N2
    end

    OUT(["Trade Setup\nML-grounded bias\nentry · tp · sl · reasoning"])

    BACKEND -->|"indicators"| INFERENCE -->|"ml_signal"| AGENT --> OUT
```

This would make the LSTM a live decision-support layer between the scanner and the LLM setup generator — combining the scanner's real-time market awareness, the LSTM's learned pattern recognition, and the LLM's reasoning and target generation.

### Implementation Steps (Planned)

1. **Add `ml_signal` to `TradeState`** in `agent/graph.py`
   ```python
   class TradeState(TypedDict):
       ...
       ml_signal: Optional[Dict[str, Any]]  # {"bias": "LONG", "confidence": 8}
   ```

2. **Create `ml_signal_node()`** in `agent/graph.py`
   ```python
   async def ml_signal_node(state: TradeState) -> dict:
       # Load LSTM model (cached at module level after first load)
       # Run inference on state["indicators"]
       # Return {"ml_signal": {"bias": "LONG", "confidence": 8}}
   ```

3. **Update generator prompt** to include ML signal as prior

4. **Update evaluator** to flag LLM vs ML bias disagreement as an issue

5. **Update `langgraph.json`** to add model path env var:
   ```json
   {"env": {"LSTM_MODEL_PATH": "optimization/results/lstm_final.pt"}}
   ```

### Research Value (Thesis)

This integration enables a key experiment for the thesis:

| Setup | Description |
|-------|-------------|
| Baseline | Zero-shot LLM alone (current) |
| ML-grounded | LSTM signal → LLM generator |
| Fine-tuned | QLoRA Qwen 2.5 7B alone |
| Hybrid | LSTM signal → Fine-tuned LLM |

The hypothesis: **ML-grounded LLM outperforms both zero-shot LLM and standalone ML** because it combines learned pattern recognition with explainable reasoning and dynamic target generation.

---

*Last updated: 2026-05-10*
*Dataset: 56,161 samples | 12 symbols | 18 months | 3 TFs*
*Best model: LSTM (85.5% test accuracy, Profit Factor 7.02)*
