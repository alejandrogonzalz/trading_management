# LangGraph Agent, Backtest Framework & Optimization

## Overview
Standalone Python research system (separate from Docker backend) for:
1. **LangGraph Agent** — 3-node DAG for intelligent trade setup generation
2. **Backtest Framework** — Evaluate LLM vs ML prediction quality against historical data
3. **Optimization Pipeline** — Hyperparameter search for XGBoost, RF, LSTM, QLoRA
4. **Dataset** — 56,161 labeled samples, 12 symbols, 18 months, 3 timeframes

**Entry Point**: `langgraph/cli.py`
**Agent API**: FastAPI on port 2024 (`langgraph/agent/main.py`)
**Environment**: `langgraph/.venv/` (Python 3.11) — activate with `source .venv/bin/activate`

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

### Shared Prompts (`langgraph/agent/prompts.py`)
Single source of truth for prompts used across the agent graph, backtest runner, and fine-tuning export:
```python
build_system_prompt(mode: Literal["SPOT", "FUTURES"]) -> str
build_user_prompt(symbol: str, indicators: Dict[str, Any]) -> str
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

### Directory Structure
```
backtest/
├── config.py              # DEFAULT_SYMBOLS, DEFAULT_TIMEFRAMES, DEFAULT_MONTHS
├── pipeline.py            # DataPipeline class — orchestrates fetch → indicators → label
├── export.py              # Fine-tuning data export (chat-format JSONL)
├── ingestion/
│   ├── fetcher.py         # Binance candle download (async, auto-paginated, concurrent TFs)
│   ├── indicators.py      # TA-Lib batch indicator calculation + multi-TF alignment
│   └── labeler.py         # Hindsight labeling algorithm
├── models/
│   ├── features.py        # Feature extraction + dataset utilities (_load_dataset, _temporal_split)
│   ├── sklearn_models.py  # XGBoostPredictor, RandomForestPredictor, get_predictor()
│   └── lstm.py            # LSTMPredictor, _LSTMNet
└── evaluation/
    ├── metrics.py          # All metric functions (accuracy, win rate, Sharpe, drawdown, etc.)
    ├── simulate.py         # _parse_prediction(), simulate_trade()
    ├── runner.py           # LLMBacktestRunner, MLBacktestRunner (class-based)
    ├── compare.py          # Side-by-side comparison of two result JSONs
    └── report.py           # Terminal tables + matplotlib equity curves
```

### Data Pipeline Flow
```
DataPipeline.fetch()        → ingestion/fetcher.py   → data/candles/{SYM}_{TF}.json
DataPipeline.build_dataset()→ ingestion/indicators.py → multi-TF indicator points
                            → ingestion/labeler.py   → data/labeled/dataset.jsonl
LLMBacktestRunner.run()     → evaluation/runner.py   → data/results/{tag}.json
MLBacktestRunner.run()      → evaluation/runner.py   → data/results/{tag}.json
```

### `ingestion/fetcher.py`
- `fetch_candles(symbol, interval, start_date, end_date)` — HTTP GET `/api/v3/klines`, auto-paginates 1000-candle chunks
- `fetch_multi_tf_candles(symbol, timeframes)` — concurrent with `asyncio.Semaphore(4)` + exponential backoff
- `save_candles()` / `load_candles()` → `data/candles/{SYMBOL}_{interval}.json`
- Default range: 18 months

### `ingestion/indicators.py`
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
# For each 1h candle, aligns the most recent 4h/1d/etc indicator snapshot
# Returns: [{"timestamp", "indicators": {"1h": {...}, "4h": {...}, "1d": {...}}, "atr_raw"}, ...]
```

### `ingestion/labeler.py` — Hindsight Labeling Algorithm
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

### `models/features.py` — Feature Extraction
```python
extract_features(indicators, timeframes=None) -> list[float]
# 9 features per TF (7 numeric + 2 encoded categorical) + 3 cross-TF features
# Dynamic size: n_TFs × 9 + 3 (inferred from data, consistent via _timeframes stored in model)
```

Private utilities also used by `optimization/`:
- `_load_dataset(path)` — reads JSONL, returns list of dicts
- `_temporal_split(samples, train_frac, val_frac)` — 70/15/15 split (never shuffled)
- `_samples_to_xy(samples, timeframes)` — returns (X: np.ndarray, y: np.ndarray)
- `_sorted_timeframes(indicators)` — sorts TF keys by duration (shortest first)
- `_infer_timeframes(samples)` — detects TF order from first valid multi-TF sample

### `models/sklearn_models.py` / `models/lstm.py` — ML Predictors

Feature vector: `n_timeframes × 9 + 3` (dynamic size stored with model)

**XGBoostPredictor**: trains with early stopping on val set, stores `.meta.pkl` alongside the model file
**RandomForestPredictor**: pickled as `{"model": ..., "timeframes": ...}` dict
**LSTMPredictor**:
- Architecture: LSTM(input_size, hidden_size, num_layers) → Linear(hidden_size, 2)
- Sequences of 10 steps (last 10 candles × features), zero-padded for early samples
- Early stopping (patience=10), Adam optimizer, z-score normalization stored with model
- `predict_sequence(tensor)` for pre-built sequence batches (used in ML backtest)

Temporal split: 70% train / 15% val / 15% test (never shuffled)

### `evaluation/runner.py` — Backtest Runners

**LLMBacktestRunner** — calls LLM for each sample, parses JSON response, simulates trade:
```python
runner = LLMBacktestRunner(dataset_path="...", provider="groq", tag="zero-shot-groq")
result = asyncio.run(runner.run())
```

**MLBacktestRunner** — trains model on train+val splits, evaluates on test split:
```python
runner = MLBacktestRunner(dataset_path="...", model_type="lstm", tag="ml-lstm", serialize=True)
result = runner.run()
```

Both runners write results to `data/results/{tag}.json` in the same schema:
```python
{
    "tag", "provider", "model", "dataset",
    "total_samples", "successful_predictions", "errors", "elapsed_seconds",
    "metrics": {...},
    "predictions": [...], "actuals": [...],
    "trade_results": [{"outcome": "WIN"|"LOSS"|"TIMEOUT", "pnl_pct": float}],
}
```

### `evaluation/metrics.py` — All Metrics
| Metric | Function |
|--------|----------|
| Direction accuracy | `direction_accuracy()` |
| Precision/Recall/F1 | `precision_recall_f1()` per class |
| Win rate | `win_rate()` — fraction reaching TP |
| Profit factor | `profit_factor()` — gross gain / gross loss |
| Avg win / avg loss | `avg_win_loss()` |
| Sharpe ratio | `sharpe_ratio()` — annualized |
| Max drawdown | `max_drawdown()` — equity curve |
| Confidence calibration | `confidence_calibration()` — binned confidence vs actual accuracy |

### `export.py` — Fine-tuning Data Export
Converts labeled dataset → chat JSONL for LLM fine-tuning:
```jsonl
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "Symbol: BTCUSDT\nIndicators: {..."}, {"role": "assistant", "content": "{\"bias\": \"LONG\", ...}"}]}
```
Temporal split: train/val/test.jsonl

### `pipeline.py` — DataPipeline Class
```python
pipe = DataPipeline(symbols=["BTCUSDT"], timeframes=["15m","1h","4h","1d"], base_tf="1h")
asyncio.run(pipe.fetch())   # download candles
pipe.build_dataset()         # indicators + labeling → data/labeled/dataset.jsonl
# Or combined:
pipe.run()
```

---

## Optimization Pipeline (`langgraph/optimization/`)

### `optimize.py`
Three search strategies:
1. **Sklearn (XGBoost, RF)**: `GridSearchCV` or `RandomizedSearchCV` + `TimeSeriesSplit(n_splits=3)`
2. **LSTM**: Manual `itertools.product` loop, random sampling if too many combos, early stopping (patience=10)
3. **QLoRA**: Prints recommended configs; manual training with Unsloth + GPU

Config YAMLs:
- `configs/xgboost.yaml` — random search, L1/L2 regularization round 2
- `configs/random_forest.yaml` — 100 random iterations
- `configs/lstm.yaml` — 30 random iterations
- `configs/qlora.yaml` — 5 recommended QLoRA configs

Output: `results/{model}_optimization.json` with best_params, all_results, scores, timing

`MLBacktestRunner` auto-loads `best_params` from this file when `--serialize` is not used.

### QLoRA Fine-tuning (`optimization/qlora/`)
```
qlora/
├── train_qlora.py         # Main training script (Unsloth + SFTTrainer)
├── run_3configs.sh        # Sequential wrapper for 3 hyperparameter configs
├── sagemaker_setup.sh     # SageMaker environment setup (venv, deps, GPU check)
├── logs/                  # Training logs per config
└── results/               # qlora_config*.json result files
```

Key details:
- Base model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` (4-bit quantized, LoRA in bf16)
- Trains on chat-format JSONL exported by `backtest/export.py`
- TRL version detection: SFTConfig for TRL>=0.12, legacy TrainingArguments otherwise
- Evaluates on same temporal test split as LSTM/XGBoost (paired comparison)
- Outputs: adapters, GGUF (Q8_0), and result JSON with predictions + trade metrics
- See `.claude/rules/qlora-training.md` for full hyperparameter reference

### `analyze_results.py`
Loads result JSONs from 3-4 models → comparison table + matplotlib charts

### `stats_tests.py` — Statistical significance (thesis validation)
Pure-Python McNemar (exact binomial) + paired t-test (normal approx). Aligns two
result JSONs by `sample_keys` (intersection) or positional fallback. All runners
(`LLMBacktestRunner`, `MLBacktestRunner`, `QLoRATrainer.evaluate`) now emit
`sample_keys` + per-sample `predictions`/`actuals`/`trade_results` so the
comparison is paired and valid. **All models share the same temporal split**
(`features._temporal_split`, no sort) — `export_training_data` was unified to use
it so the fine-tuned model's test set matches LSTM/XGBoost.

---

## CLI Commands (`langgraph/cli.py`)
```bash
# Activate venv first: cd langgraph && source .venv/bin/activate

python -m cli fetch-candles --symbols BTCUSDT,ETHUSDT --interval 1h --months 18 --timeframes "1h,4h,1d"
python -m cli prepare-dataset --symbols BTCUSDT,ETHUSDT --interval 1h --timeframes "1h,4h,1d"
python -m cli run-backtest --dataset backtest/data/labeled/dataset.jsonl --provider mock
python -m cli run-backtest --dataset backtest/data/labeled/dataset.jsonl --provider ollama
python -m cli train-ml --model xgboost --dataset backtest/data/labeled/dataset.jsonl
python -m cli train-ml --model lstm --dataset backtest/data/labeled/dataset.jsonl --serialize
python -m cli export-training-data --dataset backtest/data/labeled/dataset.jsonl --output training_data/
python -m cli compare --baseline results/baseline.json --candidate results/ml-xgboost.json
python -m cli compare-stats --a optimization/results/qlora_optimization.json --b backtest/data/results/ml-lstm.json
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
DEFAULT_TIMEFRAMES = ["15m", "1h", "4h", "1d", "1w"]   # 5 TFs (base: 1h)
DEFAULT_MONTHS = 18
```

Data files: `langgraph/backtest/data/`
- `candles/` — JSON files per symbol × TF (e.g. `BTCUSDT_1h.json`)
- `labeled/dataset.jsonl` — 56,161 labeled samples
- `results/` — backtest result JSONs for each model/provider
- `models/` — serialized trained models

---

## Tests (`langgraph/tests/`)
```
tests/
├── conftest.py                    # Adds langgraph/ root to sys.path
├── backtest/
│   └── test_backtest.py           # 35 unit tests (prompts, features, predictors, metrics, pipeline)
└── optimization/
    └── test_optimization.py       # Optimization pipeline tests
```

Run: `cd langgraph && source .venv/bin/activate && python -m pytest tests/backtest/ -v`

---

## Current Status (2026-06-13)

### Completed ✅
- LangGraph agent (3 nodes) — production-ready, running on port 2024
- Backtest framework — full CLI with 6 commands, refactored into clean subpackages
- Shared prompts (`agent/prompts.py`) — single source used by graph, backtest, and fine-tuning export
- Dataset: 56,161 samples (18 months, 12 symbols, 3 TFs)
- ML optimization complete (Avance 4 + 5):
  - **Bagging-LSTM: 83.37% test acc** (5 bags, hidden=32, layers=3, seq=5) — Avance 5 winner
  - LSTM individual: 81.5% test acc (hidden=32, layers=3, seq_len=5, dropout=0.1, lr=0.001)
  - Blending: 81.89% test acc (heterogeneous ensemble)
  - SVM (RBF): 70.8% val acc
  - XGBoost: 66.6% test acc (tuned with L1/L2 regularization)
- DVC initialized — dataset, candles, models tracked in S3 (`s3://trading-management-dvc/`)
- QLoRA config 1 trained on SageMaker (92% direction accuracy):
  - lr=2e-5, rank=16, alpha=32, epochs=2, batch=2, grad_accum=8
  - Model backed up: `s3://trading-management-dvc/models/qlora_config1/`
  - Trade simulation eval re-running with candles

### Pending ⏳
1. QLoRA configs 2-3 (rank=8/lr=5e-5/epochs=3, rank=32/lr=1e-5/epochs=2)
2. Zero-shot LLM backtest (Groq or Ollama)
3. Statistical comparison: McNemar test + paired t-test across all models
4. Integrate best fine-tuned model into LangGraph (GGUF → Ollama)
5. Ensemble LSTM+LLM (if time permits)
6. Thesis presentation + video demo (deadline ~2026-06-26)

### Research Context (Master's Thesis)

**Title**: Fine-Tuning de un LLM para Analisis Tecnico y Clasificacion de Setups en Criptomonedas
**Program**: MNA (Maestria en IA Aplicada), Tecnologico de Monterrey, Abr-Jun 2026
**Hypothesis**: A fine-tuned LLM will outperform the zero-shot baseline in direction accuracy and trade quality

**Objectives**:
1. Build labeled dataset (56K+ samples) from Binance historical candles + hindsight labeling
2. Fine-tune Qwen 2.5 7B via QLoRA (local + SageMaker) — two infra approaches
3. Evaluate 5 models: zero-shot LLM, fine-tuned LLM, XGBoost, SVM, LSTM
4. Statistical validation: McNemar test + paired t-test (paired by sample_keys)
5. Integrate best model into LangGraph agent
6. (Optional) Ensemble LSTM + LLM — combines ML accuracy with LLM explainability

**Success criteria**: at least one fine-tuned model beats zero-shot in accuracy; profit factor > 1.0; documented comparison with quantitative metrics. A negative result is academically valid if causes are analyzed.

**Deliverables**: data pipeline, dataset, fine-tuned models (local + cloud), comparative evaluation, LangGraph integration, technical report, video demo, presentation

**Evaluation metrics**: direction accuracy, precision/recall/F1, win rate, profit factor, Sharpe ratio, max drawdown, confidence calibration, latency, training cost

Full proposal: `trading_management_docs/fine-tunning/propuesta/Definicion_Del_Proyecto_MNA.md`

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
