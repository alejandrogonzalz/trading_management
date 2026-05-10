# LangGraph Agent, Backtest Framework & Optimization

## Overview
Standalone Python research system (separate from Docker backend) for:
1. **LangGraph Agent** — 3-node DAG for intelligent trade setup generation
2. **Backtest Framework** — Evaluate LLM vs ML prediction quality against historical data
3. **Optimization Pipeline** — Hyperparameter search for XGBoost, RF, LSTM
4. **Dataset** — 56,161 labeled samples, 12 symbols, 18 months, 3 timeframes

**Entry Point**: `langgraph/cli.py`
**Agent API**: FastAPI on port 2024 (`langgraph/agent/main.py`)
**Environment**: conda env `trading` (Python 3.11)

---

## Agent — 3-Node LangGraph DAG (`langgraph/agent/graph.py`)

### TradeState (TypedDict)
```python
class TradeState:
    run_id: str                           # UUID per execution
    symbol: str                           # e.g. "BTCUSDT"
    mode: Literal["SPOT", "FUTURES"]
    indicators: Dict[str, Any]            # {"1h": {...}, "4h": {...}, "1d": {...}}
    original: Optional[Dict]              # Generator output
    evaluation: Optional[Dict]            # Evaluator output
    optimized: Optional[Dict]             # Optimizer output (if triggered)
    issues: List[str]                     # Problems detected by evaluator
    needs_optimization: bool              # Conditional edge flag
    audit_trail: List[Dict]               # Per-node timing in ms
```

### Node 1: `generator_node()` (async, LLM)
- Calls LLM with system prompt defining SPOT/FUTURES mode + trading rules
- Output JSON: `{"bias", "entry", "tp", "sl", "leverage", "reasoning", "quality"}`
- Safety: Forces `bias="LONG"` and `leverage=null` when mode=SPOT (hallucination guard)
- Retries JSON parse 2x before fallback

### Node 2: `evaluator_node()` (sync, deterministic — NO LLM)
- Validates setup quantitatively:
  - `tp > entry > sl` for LONG / `tp < entry < sl` for SHORT
  - RSI > 75 (LONG) or < 25 (SHORT) → issue "overbought/oversold"
  - ATR ratio > 2.5 → issue "high volatility"
  - FUTURES: SL distance < 70% of liquidation distance → issue "liquidation risk"
  - Multi-TF alignment: heatmaps from 1h/4h/1d must agree with bias
- Output: `{"confidence", "quant_confidence", "macro_alignment", "liquidation_safe", "risk_pct", "liq_pct", "safety_margin", "rr", "issues"}`
- If issues detected → sets `needs_optimization=True`

### Node 3: `optimizer_node()` (async, LLM — conditional)
- Only runs if `needs_optimization == True`
- Prompt: "Risk Manager" persona — refine setup given issues + indicators
- Output: original JSON + `"changes": ["Reduced leverage from 5x to 3x", ...]`
- Falls back to `None` if LLM fails

### Graph Edges
```
generator → evaluator → (conditional) → optimizer → END
                     └─ (no issues)  → END
```

### REST API (`langgraph/agent/main.py`)
```
POST /analyze
Body: {"symbol": str, "mode": "SPOT"|"FUTURES", "indicators": {tf: {...}}}
Response: {run_id, symbol, mode, original, evaluation, optimized, issues, audit_trail}
```

---

## LLM Factory (`langgraph/agent/llm_factory.py`)

Supported providers (via `LLM_PROVIDER` env var):
| Provider | Class | Notes |
|----------|-------|-------|
| `ollama` | `OllamaProvider` | Default, local Qwen 2.5 14B. Retry 3x with exponential backoff |
| `google` | `GoogleProvider` | Requires `GOOGLE_API_KEY` |
| `bedrock` | `BedrockProvider` | Requires `AWS_REGION` |
| `openai` | `OpenAIProvider` | Requires `OPENAI_API_KEY` |
| `groq`, `deepseek`, `together`, `fireworks` | `OpenAICompatibleProvider` | Auto-maps base URL |
| `mock` | `MockProvider` | Deterministic fake — for tests without API calls |

---

## Backtest Framework (`langgraph/backtest/`)

### Pipeline
```
1. fetch_candles.py   → Binance klines, 12 symbols × 3 TFs × 18 months → JSON
2. calculate_indicators.py → TA-Lib batch → multi-TF indicator points
3. label_data.py      → Hindsight labeling → 56,161 samples JSONL
4. run_backtest.py    → LLM predictions → trade simulation → metrics
   OR run_ml_backtest.py → ML predictions → same metrics
5. compare.py         → Side-by-side comparison of two result JSONs
6. report.py          → Terminal table output
```

### `fetch_candles.py`
- `fetch_candles(symbol, interval, start_date, end_date)` — HTTP GET `/api/v3/klines`, auto-paginates 1000 candle chunks
- `fetch_multi_tf_candles(symbol, timeframes)` — parallel multi-TF
- `save_candles()` → `data/candles/{SYMBOL}_{interval}.json`
- Default range: 18 months (Nov 2024 – May 2026)

### `calculate_indicators.py`
Indicators computed per candle (TA-Lib):
| Indicator | Period | Purpose |
|-----------|--------|---------|
| EMA 20/50/200 | 20/50/200 | Trend via EMA alignment |
| ADX | 14 | Trend strength (0-100) |
| RSI | 14 | Overbought/oversold |
| MACD | 12/26/9 | Momentum |
| ATR | 14 | Volatility in price units |
| Bollinger Bands | 20, 2σ | Extremes |
| Volume SMA | 20 | Volume baseline |

Derived features:
- `heatmap`: `STRONG_BULLISH | BULLISH | NEUTRAL | BEARISH | STRONG_BEARISH`
- `structure`: `BREAKOUT | BULLISH | RANGE | BEARISH | BREAKDOWN`
- `atr_ratio`: ATR / SMA(20 ATR) — relative volatility
- `volume_ratio`: Volume / SMA(20 volume)
- `bb_pos`: Price position within Bollinger Bands (0=lower, 1=upper)

Key function:
```python
calculate_multi_tf_indicators(candles_by_tf, base_tf="1h", lookback=200)
# For each 1h candle, aligns the most recent 4h and 1d indicator snapshot
# Returns: [{"timestamp", "indicators": {"1h": {...}, "4h": {...}, "1d": {...}}, "atr_raw"}, ...]
```

### `label_data.py` — Hindsight Labeling Algorithm
1. Extract ATR from indicator point
2. Dynamic threshold: `ATR × 1.5`
3. Look 24 candles ahead
4. `max_up = (max_high - entry) / entry`, `max_down = (entry - min_low) / entry`
5. Quality filters:
   - `volume_ratio >= 0.5` (min volume)
   - `ADX >= 15` (min trend)
   - `reward / risk >= 1.0` (min R:R)
   - Anti-whipsaw: reversal in first 4 candles > threshold → discard
6. Direction:
   - `max_up > threshold AND max_up > max_down × 1.5` → **LONG** (TP = entry × (1 + max_up × 0.7), SL = entry - 1 ATR)
   - `max_down > threshold AND max_down > max_up × 1.5` → **SHORT** (inverse)
   - Ambiguous → discard
7. Confidence: 1-10 based on R:R + ADX
8. Quality: "HIGH" if R:R ≥ 2:1, else "MEDIUM"

**Dataset stats**: 56,161 samples, 49% LONG / 51% SHORT (balanced), from 250K+ raw candles

### `run_backtest.py` — LLM Backtest
```python
run_backtest(dataset_path, provider, tag, sample_limit)
# For each sample: build LLM prompt → parse JSON → simulate trade (check 24 future candles for TP/SL hit) → PnL
```
Output JSON:
```python
{
    "tag", "provider", "model", "dataset",
    "total_samples", "successful_predictions", "errors", "elapsed_seconds",
    "metrics": {...},
    "predictions": [...], "actuals": [...],
    "trade_results": [{"outcome": "WIN"|"LOSS"|"TIMEOUT", "pnl_pct": float}],
    "audit_trail": [...]
}
```

### `metrics.py` — All Metrics
| Metric | Function |
|--------|----------|
| Direction accuracy | `direction_accuracy()` |
| Precision/Recall/F1 | `precision_recall_f1()` per class |
| Win rate | `win_rate()` — fraction reaching TP |
| Profit factor | `profit_factor()` — gross gain / gross loss |
| Avg win / avg loss | `avg_win()`, `avg_loss()` |
| Sharpe ratio | `sharpe_ratio()` — annualized |
| Max drawdown | `max_drawdown()` — equity curve |
| Confidence calibration | `confidence_calibration()` — binned confidence vs actual accuracy |

### `ml_models.py` — ML Predictors

Feature extraction: 30 features per sample
- Multi-TF indicators (3 TFs × 7 numeric + 2 categorical) = 27 base
- Cross-TF: TF agreement, RSI divergence, volume spread = 3 extra

**XGBoostPredictor**: GridSearchCV + TimeSeriesSplit(n_splits=3)
**RandomForestPredictor**: RandomizedSearchCV (100 iters)
**LSTMPredictor**:
- Architecture: LSTM(input_size, hidden_size, num_layers) → Linear(hidden_size, 2)
- Sequences of 10 steps (last 10 candles × features)
- Early stopping (patience=10), Adam optimizer
- Normalization: mean/std of training set stored with model

Temporal split: 70% train / 15% val / 15% test (never shuffled)

### `export_training_data.py` — Fine-tuning Data Export
Converts labeled dataset → chat JSONL for LLM fine-tuning:
```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "Symbol: BTCUSDT\nIndicators: {..."}, {"role": "assistant", "content": "{\"bias\": \"LONG\", ...}"}]}
```
Temporal split: train/val/test.jsonl

---

## Optimization Pipeline (`langgraph/optimization/`)

### `optimize.py`
Three search strategies:
1. **Sklearn (XGBoost, RF)**: `GridSearchCV` or `RandomizedSearchCV` + `TimeSeriesSplit(n_splits=3)`
2. **LSTM**: Manual `itertools.product` loop, random sampling if too many combos, early stopping (patience=10)
3. **QLoRA**: Prints recommended configs; manual training with Unsloth + GPU

Config YAMLs:
- `configs/xgboost.yaml` — 576 combinations (grid)
- `configs/random_forest.yaml` — 100 random iterations
- `configs/lstm.yaml` — 30 random iterations
- `configs/qlora.yaml` — 5 recommended QLoRA configs

Output: `results/{model}_optimization.json` with best_params, all_results, scores, timing

### `analyze_results.py`
Loads result JSONs from 3-4 models → comparison table + matplotlib charts

---

## CLI Commands (`langgraph/cli.py`)
```bash
# Activate conda env first: conda activate trading

python -m cli fetch-candles --symbols BTCUSDT,ETHUSDT --interval 1h --months 18 --timeframes "1h,4h,1d"
python -m cli prepare-dataset --symbols BTCUSDT,ETHUSDT --interval 1h --timeframes "1h,4h,1d"
python -m cli run-backtest --dataset backtest/data/labeled/dataset.jsonl --provider mock
python -m cli run-backtest --dataset backtest/data/labeled/dataset.jsonl --provider ollama
python -m cli train-ml --model xgboost --dataset backtest/data/labeled/dataset.jsonl
python -m cli train-ml --model lstm --dataset backtest/data/labeled/dataset.jsonl
python -m cli export-training-data --dataset backtest/data/labeled/dataset.jsonl --output training_data/
python -m cli compare --baseline results/baseline.json --candidate results/ml-xgboost.json
```

---

## LangGraph Deploy Config (`langgraph/langgraph.json`)
```json
{
    "dependencies": ["."],
    "graphs": {"trading_agent": "./agent/graph.py:graph"},
    "env": {"OLLAMA_BASE_URL": "http://localhost:11434"}
}
```

---

## Dataset & Symbols
```python
DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "BNBUSDT",         # Large cap
    "SOLUSDT", "XRPUSDT", "ADAUSDT", "AVAXUSDT", "DOTUSDT",  # Mid cap
    "DOGEUSDT", "LINKUSDT", "MATICUSDT", "NEARUSDT"           # Volatile
]
DEFAULT_TIMEFRAMES = ["1h", "4h", "1d"]
DEFAULT_MONTHS = 18
```

Data files: `langgraph/backtest/data/`
- `candles/` — 36 JSON files (12 symbols × 3 TFs)
- `labeled/dataset.jsonl` — 56,161 labeled samples
- `results/` — backtest result JSONs for each model/provider

---

## Current Status (2026-05-09)

### Completed ✅
- LangGraph agent (3 nodes) — production-ready, running on port 2024
- Backtest framework — full CLI with 6 commands
- Dataset: 56,161 samples (18 months, 12 symbols, 3 TFs)
- Optimization round 1 complete:
  - **LSTM: 84.29%** accuracy (winner, CPU, 73.5 min training)
  - XGBoost: 76.79% (overfitting gap: 0.187)
  - Random Forest: 74.89% (overfitting gap: 0.203)

### Pending ⏳
1. Evaluate LSTM winner on test set (~8,425 unseen samples)
2. Serialize LSTM with `torch.save()` for deployment
3. Second optimization round for XGB/RF (L1/L2 regularization)
4. Initialize DVC for model versioning
5. Refactor `optimization/` to class-based pipeline
6. QLoRA fine-tuning of Qwen 2.5 7B (thesis requirement)

### Research Context
This is part of a **master's thesis** research project:
> "Can fine-tuning significantly improve LLM accuracy for crypto technical analysis while maintaining explainable reasoning?"

5 models to compare: zero-shot LLM, fine-tuned LLM (QLoRA), XGBoost, Random Forest, LSTM
Statistical validation: McNemar test + paired t-test

---

## Key Dependencies
| Library | Use |
|---------|-----|
| langgraph | Agent DAG framework |
| langchain-* | LLM provider integrations |
| fastapi + uvicorn | Agent REST API |
| ta-lib | Technical indicator computation |
| xgboost | XGBoost classifier |
| scikit-learn | RF, GridSearchCV, TimeSeriesSplit |
| torch | LSTM training |
| pandas + numpy | Data manipulation |
| httpx | Async Binance API calls |
| pyyaml | Optimization configs |
| tqdm | Progress bars |
| matplotlib | Results visualization |
