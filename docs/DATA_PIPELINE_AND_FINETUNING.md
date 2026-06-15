# Data Pipeline & QLoRA Fine-Tuning — How It Works

Explains the two key questions: **where does the training data come from?** and
**what does fine-tuning actually do to the model?**

---

## Part 1 — Building the Dataset

The dataset starts from raw Binance candles and ends as `dataset.jsonl` — 56,161
labeled trading setups ready to train on. Five stages:

```
Binance API
    │
    ▼
[1] Candle fetch       → data/candles/{SYM}_{TF}.json
    │
    ▼
[2] Indicator calc     → multi-TF indicator snapshots (1h / 4h / 1d)
    │
    ▼
[3] Hindsight label    → LONG / SHORT / discard (per 1h candle)
    │
    ▼
[4] Temporal split     → train (70%) / val (15%) / test (15%)
    │
    ▼
[5] Chat export        → messages JSONL (system + user + assistant triples)
```

---

### Stage 1 — Candle Fetch (`backtest/ingestion/fetcher.py`)

Downloads 18 months of OHLCV bars from `GET /api/v3/klines`.

```python
# Fetches all candles for a symbol+timeframe, paginating 1000-bar chunks
fetch_candles(symbol="BTCUSDT", interval="1h", start_date=..., end_date=...)
```

For each of the 12 symbols × 5 timeframes (15m / 1h / 4h / 1d / 1w), the result
is cached as `data/candles/BTCUSDT_1h.json`. Total: ~250,000 raw candles.

All fetches run concurrently (`asyncio.Semaphore(4)`) with exponential backoff.

---

### Stage 2 — Indicator Calculation (`backtest/ingestion/indicators.py`)

For every 1h candle, computes all indicators and aligns the matching 4h and 1d
snapshots to that timestamp.

**Indicators computed (via TA-Lib):**

| Indicator | Period | What it measures |
|-----------|--------|-----------------|
| EMA | 20 / 50 / 200 | Trend direction via moving average alignment |
| ADX | 14 | Trend strength (0 = flat, 100 = strong trend) |
| RSI | 14 | Momentum oscillator — overbought (>70) / oversold (<30) |
| MACD | 12/26/9 | Momentum crossover signal |
| ATR | 14 | Average true range — volatility in price units |
| Bollinger Bands | 20, 2σ | Price position relative to volatility envelope |
| Volume SMA | 20 | Baseline volume for ratio calculation |

**Derived signals computed from the raw indicators:**

- `heatmap`: `STRONG_BULLISH | BULLISH | NEUTRAL | BEARISH | STRONG_BEARISH`
  — EMA alignment (short vs long), RSI zone, MACD direction
- `structure`: `BREAKOUT | BULLISH | RANGE | BEARISH | BREAKDOWN`
  — whether price is above/below key EMAs and breaking out of range
- `atr_ratio`: ATR / rolling_mean(ATR, 20) — is volatility high or normal?
- `volume_ratio`: volume / volume_SMA_20 — is this a high-volume candle?
- `bb_pos`: (price - lower_band) / (upper_band - lower_band) — 0 = bottom of bands, 1 = top

The key function:
```python
calculate_multi_tf_indicators(candles_by_tf, base_tf="1h", lookback=200)
# For each 1h candle at timestamp T:
#   - Computes 1h indicators from the preceding 200 bars
#   - Finds the most recent 4h candle where close_time ≤ T, computes its indicators
#   - Same for 1d
# Returns a list of indicator "points" — one per 1h candle
```

Each output point looks like:
```json
{
  "timestamp": 1719000000,
  "atr_raw": 312.4,
  "indicators": {
    "1h": {"rsi": 48.3, "adx": 22.1, "ema_20": 67100, "heatmap": "NEUTRAL", ...},
    "4h": {"rsi": 55.6, "adx": 30.4, "ema_20": 66800, "heatmap": "BULLISH", ...},
    "1d": {"rsi": 61.2, "adx": 18.7, "ema_20": 65000, "heatmap": "BULLISH", ...}
  }
}
```

---

### Stage 3 — Hindsight Labeling (`backtest/ingestion/labeler.py`)

This is the core of the dataset. For each 1h candle, we look into the future (the
next 24 candles) to determine whether a trade opened at that candle would have won.

**The algorithm, step by step:**

```
Given: indicator_point at timestamp T (entry price = current close)

1. Compute threshold:
   threshold = ATR × 1.5
   (e.g. if ATR = 300 USDT, threshold = 450 USDT = ~0.67% move on BTC)

2. Look at the next 24 candles (24h forward):
   max_up   = (max_high   - entry) / entry   # best case bullish move
   max_down = (entry - min_low)    / entry   # best case bearish move

3. Quality pre-filter (discard if any fail):
   - volume_ratio >= 0.5  (not a dead-volume candle)
   - ADX >= 15            (some trend present)
   - Anti-whipsaw: if the first 4 candles reverse > threshold, discard
     (price went up then came back = no clean trade)

4. Label direction:
   LONG  if max_up   > threshold AND max_up   > max_down × 1.5
   SHORT if max_down > threshold AND max_down > max_up   × 1.5
   discard if ambiguous (both directions similar, or neither big enough)

5. Compute R:R and discard if reward/risk < 1.0

6. For LONG:
   TP = entry × (1 + max_up × 0.7)    # 70% of the actual move (conservative)
   SL = entry - 1 × ATR               # one ATR below entry

7. Confidence 1-10: based on R:R ratio + ADX strength
   Quality: "HIGH" if R:R >= 2:1, else "MEDIUM"
```

From ~250,000 raw candles, ~56,161 pass all filters (≈22% keep rate). The
discard rate is intentional — ambiguous or choppy candles make poor training
examples and would teach the model to pick random directions.

Output: each labeled sample in `dataset.jsonl`:
```json
{
  "symbol": "BTCUSDT",
  "timestamp": 1719000000,
  "label": "LONG",
  "entry": 67420.5,
  "tp": 68890.2,
  "sl": 67108.1,
  "confidence": 7,
  "quality": "HIGH",
  "indicators": { "1h": {...}, "4h": {...}, "1d": {...} }
}
```

---

### Stage 4 — Temporal Split (`backtest/models/features.py: _temporal_split`)

The 56,161 samples are split into train / val / test **by time**, never shuffled.

```
Sort ALL 56,161 samples globally by timestamp (ascending)
             │
             ▼
[Oldest ─────────────────────────────────────── Newest]
│    70% = 39,313 train    │ embargo │  15% = 8,424 val  │ embargo │  15% = 8,424 test  │
```

**Why the embargo?** The labeling algorithm looks 24 candles forward. A sample
at timestamp T uses price data up to T+24h. If T is near the train/val boundary,
its label bleeds into the val period. The embargo (24 bars × 1h = 24h gap) removes
any sample whose lookahead window crosses the boundary — preventing label leakage.

**Why global sort instead of per-symbol?** `dataset.jsonl` is grouped by symbol
(all BTCUSDT rows first, then all ETHUSDT rows, etc.). A positional split would
train on all of 2023 BTC + early 2024 ETH and test on late 2024 BTC — the model
never sees recent BTC during training. This was the original leakage bug. Global
sort interleaves all symbols chronologically, so the split is a true time-based
holdout.

---

### Stage 5 — Chat Export (`backtest/export.py`)

Each labeled sample becomes a 3-message conversation in chat format — exactly how
the model was instruction-tuned, and exactly how it will be called at inference.

```python
# System message: the trading analyst persona + output format rules
system = build_system_prompt(mode="SPOT")

# User message: the indicator snapshot for this candle
user = build_user_prompt(symbol="BTCUSDT", indicators={...})

# Assistant message: the correct answer (from hindsight labeling)
assistant = json.dumps({
    "bias": "LONG",
    "entry": 67420.5,
    "tp": 68890.2,
    "sl": 67108.1,
    "reasoning": "RSI at 48 with ADX 22 and BULLISH 4h structure...",
    "confidence": 7
})
```

Stored as JSONL, one JSON object per line:
```json
{"messages": [
  {"role": "system",    "content": "You are a Senior Technical Analyst..."},
  {"role": "user",      "content": "Symbol: BTCUSDT\nIndicators:\n1h: RSI=48.3, ADX=22.1..."},
  {"role": "assistant", "content": "{\"bias\": \"LONG\", \"entry\": 67420.5, ...}"}
]}
```

The train split (39,313 samples) feeds the SFTTrainer. Val split (8,424 samples)
is used for early stopping. Test split (8,424 samples) is the held-out evaluation
set — none of these were seen during training.

---

## Part 2 — How Fine-Tuning Creates Adapter Weights

### The frozen base model

Qwen 2.5 7B has 7 billion parameters. We **never change them**. They are loaded
in 4-bit NF4 quantization (the "Q" in QLoRA), which compresses each weight from
16 bits to 4 bits:

```
Normal storage:  7B weights × 16 bits = ~14 GB VRAM
4-bit storage:   7B weights × 4 bits  = ~3.5 GB VRAM
```

The base model already knows how to read, reason, and format JSON. It was
pre-trained on trillions of tokens of text and instruction-tuned by Alibaba.
We don't need to teach it language — we need to teach it *our task*.

---

### The LoRA adapters (what actually trains)

LoRA (Low-Rank Adaptation) injects small trainable matrices into the model.
For each target layer weight matrix `W`, it adds a correction `ΔW`:

```
At inference:  output = (W + ΔW) × input

Where:
  W  = frozen base weight (large, 4-bit, never changes)
  ΔW = A × B  (two small trainable matrices)

  A shape: [layer_dim × rank]   e.g. [4096 × 16]
  B shape: [rank × layer_dim]   e.g. [16 × 4096]
```

The rank (16 in our config) is the bottleneck. Instead of learning a full
`[4096 × 4096]` correction (16M new params), we learn two matrices whose
product is that correction — only 2 × (4096 × 16) = 131,072 params per layer.

**Target modules** (7 per transformer layer, 28 layers = 196 adapter pairs):
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj` — which tokens to attend to
- MLP: `gate_proj`, `up_proj`, `down_proj` — how to transform attended information

**Total trainable parameters: ~40 million** (0.57% of 7B).

The adapter output is scaled by `alpha / rank = 32 / 16 = 2.0` before being
added to the base weight output. This controls how much the adapters influence
the model's behavior vs the frozen base.

---

### One training step, in detail

```
Input: one chat conversation (system + user + assistant), tokenized

1. FORWARD PASS
   ─────────────
   Feed all ~900 tokens through 28 transformer layers.
   Each layer computes: attention + MLP, using (W + A×B) as the effective weight.
   
   The model produces a probability distribution over vocabulary tokens at each
   position — i.e., "what is the next token?"

2. LOSS COMPUTATION (cross-entropy, assistant tokens only)
   ────────────────────────────────────────────────────────
   We only care about the assistant's answer tokens:
     '{"bias": "LONG", "entry": 67420.5, ...}'
   
   For each of those tokens, compare:
     - What the model predicted (a probability distribution over 32K vocab items)
     - What it should have predicted (the actual next token)
   
   Loss = -log(probability assigned to the correct token)
   
   High loss = model was surprised by the correct token (it predicted something else)
   Low loss  = model was confident and correct

3. BACKWARD PASS (gradient computation)
   ─────────────────────────────────────
   Compute how each trainable parameter affected the loss.
   Because W is frozen, no gradient flows into W.
   Gradients flow into A and B matrices only.
   
   The gradient for each adapter parameter is:
   "if I increase this by ε, does the loss go up or down, and by how much?"

4. WEIGHT UPDATE (Adam optimizer)
   ────────────────────────────────
   For each element of A and B across all 196 adapter pairs:
   
   param = param - lr × (gradient / (running_std_of_gradients + ε))
   
   lr = 2e-5 at peak (after 50-step warmup), decaying to ~0 via cosine schedule
   
   The Adam terms (momentum + variance estimates) make the update robust to
   gradient noise and different parameter scales.

5. Repeat for the next sample.
```

One epoch = one pass through all 39,313 training examples = ~2,457 steps at
effective batch size 16 (BATCH=8 × GRAD_ACCUM=2). Training runs 2 epochs =
~4,914 total weight updates.

---

### What the training loss tells you

In the logs you see lines like:

```
{'loss': 1.8312, 'grad_norm': 1.2341, 'learning_rate': 2e-05, 'epoch': 0.10}
{'loss': 1.2104, 'grad_norm': 0.8912, 'learning_rate': 1.8e-05, 'epoch': 0.51}
{'loss': 0.6841, 'grad_norm': 0.6103, 'learning_rate': 1.1e-05, 'epoch': 1.02}
```

| Field | What it means | Healthy range |
|-------|--------------|---------------|
| `loss` | Cross-entropy on this batch | Starts ~1.8, should drop to ~0.4–0.8 |
| `grad_norm` | Magnitude of the gradient vector | 0.3–3.0; >10 = unstable |
| `learning_rate` | Current LR after warmup + cosine decay | 2e-5 → ~0 over 2 epochs |
| `epoch` | Fraction of training complete | 0.0 → 2.0 |

A healthy run: loss drops steadily, grad_norm stays in range, no spikes. An
overfit run: training loss keeps dropping but `eval_loss` (logged every 250 steps
against the val set) stops improving or increases.

---

### What gets saved (the adapter files)

After training, the adapter weights are saved as a PEFT checkpoint:

```
backtest/data/models/qlora_cloud/
├── adapter_config.json      ← LoRA config: rank=16, alpha=32, target modules
├── adapter_model.safetensors ← The A and B matrices for all 196 adapter pairs
│                               Size: ~80 MB (vs 14 GB for the full base model)
├── tokenizer.json           ← Shared with base model
└── special_tokens_map.json
```

At inference, these adapters merge with the frozen base:

```python
# Load base model in 4-bit (3.5 GB)
model = FastLanguageModel.from_pretrained("unsloth/Qwen2.5-7B-Instruct-bnb-4bit")

# Load and merge adapters (80 MB)
model = PeftModel.from_pretrained(model, "backtest/data/models/qlora_cloud/")

# The model now behaves as if W + ΔW were the base weights
```

For deployment via Ollama, the merged model is exported as GGUF (Q4_K_M
quantization) — a single `~4.5 GB` file that llama.cpp / Ollama can serve.

---

### Why this works (conceptually)

The base model already has the computation to reason about numbers, follow
output schemas, and combine multiple signals. The adapters learn a small
"redirection" — they shift the model's internal representations toward the
vocabulary of technical indicators, the expected output keys (`bias`, `entry`,
`tp`, `sl`), and the relationship between indicator states and trade directions.

Because the training examples are correctly labeled by hindsight (we looked at
what actually happened in the next 24 candles), the adapter learns a mapping
from observable indicators to outcomes that were profitable in historical data.
Whether this generalizes out-of-sample is what the held-out test set measures.

---

## Files Referenced

| File | Role |
|------|------|
| `backtest/ingestion/fetcher.py` | Downloads candles from Binance |
| `backtest/ingestion/indicators.py` | TA-Lib indicator calculation + multi-TF alignment |
| `backtest/ingestion/labeler.py` | Hindsight labeling algorithm |
| `backtest/models/features.py` | Temporal split + feature extraction |
| `backtest/export.py` | Converts dataset.jsonl → chat-format JSONL |
| `agent/prompts.py` | System + user prompt templates (shared with training) |
| `optimization/qlora/train_qlora.py` | Loads chat JSONL, runs SFTTrainer, evaluates |
| `backtest/data/labeled/dataset.jsonl` | 56,161 labeled samples (DVC-tracked) |
| `backtest/data/candles/*.json` | Raw OHLCV candles per symbol × TF (DVC-tracked) |
| `backtest/data/models/qlora_cloud/` | Trained adapter weights + GGUF (DVC-tracked) |
