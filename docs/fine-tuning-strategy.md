# Fine-Tuning Strategy & Training Plan

**Date**: April 30, 2026
**Project**: Trading Management System — Maestría en IA Aplicada, Tec de Monterrey
**Author**: Alejandro González Almazán

---

## Overview

Supervised fine-tuning of Qwen 2.5 7B to specialize it for crypto technical analysis. The fine-tuned Generator node should outperform the zero-shot baseline on direction accuracy, TP/SL quality, and reasoning coherence.

**Related docs**:
- [Backtest Framework](./backtest-framework.md) — evaluation system and comparison methodology
- [Action Plan](./action-plan.md) — implementation timeline and dependencies

---

## 1. What to Fine-Tune

### Generator Only — Not the Optimizer

| Node | Fine-tune? | Rationale |
|---|---|---|
| **Generator** | ✅ Yes | Core prediction task — direction + price levels. This is what we're measuring. |
| **Evaluator** | ❌ No | Deterministic code, no LLM involved. Already good. |
| **Optimizer** | ❌ No | Corrective layer that fires ~30-40% of the time. Low ROI — if Generator improves, Optimizer fires less. |

Both Generator and Optimizer should use the **same fine-tuned model weights**. The prompts differentiate their behavior. A model fine-tuned on generator-style examples will also be better at optimization because it understands the domain vocabulary.

### Model Choice: Qwen 2.5 7B Instruct

| Factor | 7B | 14B |
|---|---|---|
| QLoRA VRAM | ~8 GB | ~16 GB |
| Training speed | 2× faster | Baseline |
| Inference (Ollama) | ~2s | ~4s |
| JSON output quality | Excellent | Slightly better |
| Overfitting risk (3-5K examples) | Lower | Higher |

The 7B model is the sweet spot: fits in GPU for both training and inference, fast enough for real-time scanning (20 pairs), and less prone to overfitting on a small dataset.

### Method: QLoRA (4-bit Quantization + LoRA Adapters)

- Full fine-tuning of 7B requires 56+ GB VRAM — impractical locally
- QLoRA trains ~0.5% of parameters while base model stays frozen in 4-bit
- Quality difference vs full fine-tuning is negligible for 3K-10K examples
- Academic bonus: compare QLoRA local vs Together AI full fine-tuning

---

## 2. Training Data Format

### Exact JSONL Structure

Training examples **MUST** match the production `generator_node()` prompt exactly. Any mismatch degrades fine-tuning effectiveness.

```jsonl
{"messages":[{"role":"system","content":"You are a Senior Technical Analyst for a FUTURES (Long/Short, Leverage 1-50x) trading system.\nAnalyze the provided multi-timeframe indicators and generate a high-confluence trade setup.\n\nRULES:\n- MODE: FUTURES\n- Bias can be LONG or SHORT. Recommend leverage (1-50) based on volatility.\n- Output MUST be valid JSON."},{"role":"user","content":"Symbol: BTCUSDT\nIndicators: {\"1h\":{\"price\":67234.5,\"heatmap\":\"STRONG_BULLISH\",\"structure\":\"BREAKOUT\",\"rsi\":62.3,\"macd_hist\":125.4,\"adx\":35.2,\"volume_ratio\":2.1,\"atr_ratio\":1.35,\"bb_pos\":0.78},\"4h\":{...},\"1d\":{...}}\n\nReturn valid JSON with these exact fields:\n{\"bias\": \"LONG\" or \"SHORT\", \"entry\": float, \"tp\": float, \"sl\": float, \"leverage\": integer or null, \"reasoning\": \"2 sentences max\", \"quality\": \"HIGH\", \"MEDIUM\", or \"LOW\"}"},{"role":"assistant","content":"{\"bias\":\"LONG\",\"entry\":67234.5,\"tp\":69100.0,\"sl\":66400.0,\"leverage\":5,\"reasoning\":\"Strong bullish breakout on 1h confirmed by 4h uptrend. ADX 35 confirms trend strength; conservative leverage due to neutral daily structure.\",\"quality\":\"HIGH\"}"}]}
```

### Reasoning Generation

Generate reasoning programmatically — don't write 5,000 strings by hand:

```python
def generate_reasoning(bias, indicators, label_data):
    parts = []
    ind = indicators['1h']
    if bias == "LONG":
        if ind['rsi'] < 35: parts.append(f"RSI at {ind['rsi']:.0f} indicates oversold conditions")
        if ind['structure'] == "BREAKOUT": parts.append("price breaking above recent resistance")
        if ind['adx'] > 25: parts.append(f"ADX at {ind['adx']:.0f} confirms strong trend")
        if ind['volume_ratio'] > 1.5: parts.append(f"volume {ind['volume_ratio']:.1f}x above average")
    # ... symmetric for SHORT
    return ". ".join(parts[:3]) + "."
```

---

## 3. The NEUTRAL Class Decision

**Do NOT include NEUTRAL examples in training data.**

This is the single most important design decision:

1. Markets are neutral 60-70% of the time → massive class imbalance
2. A model that learns "when in doubt, say NEUTRAL" is useless — it's the safe default requiring no skill
3. The Evaluator already handles "no trade" via low `quant_confidence` scores
4. The production scanner already filters pairs below a scoring threshold before sending to LangGraph

**Instead**: Only train on clear LONG and SHORT examples. Filter aggressively. 3,000 high-quality directional examples beats 10,000 noisy examples with 6,000 NEUTRALs.

---

## 4. Labeling Algorithm

Uses ATR-based dynamic thresholds and checks drawdown-before-profit. See [Backtest Framework](./backtest-framework.md#labeling-logic) for the core labeling function.

### Key Design Choices

- **Dynamic threshold**: `1.5 × ATR / entry_price` — adapts to each pair's volatility (1% on BTC is noise; 1% on a low-cap alt is a real move)
- **Path dependency**: Check MAE (max adverse excursion) before TP was hit. If price dropped 3% before rising 2%, that's a losing trade with any reasonable SL
- **Confidence scoring**: Based on R:R ratio, ADX strength, and volume confirmation (60-90 range)
- **Quality tiers**: HIGH (R:R > 2.5, confidence ≥ 75), MEDIUM (R:R > 1.5, confidence ≥ 65), LOW (rest)

### Quality Filters

Discard examples where:
- Movement < 1.5× ATR (noise)
- Volume ratio < 0.5 (unreliable)
- ADX < 15 (random walk)
- R:R < 1:1 (not worth trading)
- Direction reversed within 4 candles (whipsaw)
- Temporal gap < 2h from previous example of same symbol (autocorrelation)

---

## 5. Dataset Composition

| Dimension | Target |
|---|---|
| Total examples | 3,000 – 5,000 |
| LONG : SHORT | 45-55% : 45-55% |
| HIGH quality | ~20% |
| MEDIUM quality | ~50% |
| LOW quality | ~30% |
| Pairs represented | All 20+ (no single pair > 15%) |
| Timeframes | 50% from 1h, 50% from 4h |
| SPOT examples | ~40% (bias always LONG, leverage null) |
| FUTURES examples | ~60% (both directions, leverage 1-20) |

### Temporal Split (NOT Random)

```
Train:      months 1-4  (first 67%)
Validation: month 5     (next 17%)
Test:       month 6     (final 17%)
```

Random splits leak future information through autocorrelated market regimes. Temporal splits prevent this.

---

## 6. Training Pipeline

### Local: QLoRA via Unsloth

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048, load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32, lora_dropout=0.05,
)

# Train with SFTTrainer (from trl)
# Export to GGUF: model.save_pretrained_gguf("trading-analyst", tokenizer, quantization_method="q4_k_m")
# Load in Ollama: ollama create trading-analyst-local -f Modelfile
```

### Hyperparameter Recommendations

| Parameter | Range to Search | Default |
|---|---|---|
| Learning rate | [1e-4, 2e-4, 5e-4] | 2e-4 |
| LoRA rank | [8, 16, 32] | 16 |
| LoRA alpha | 2× rank | 32 |
| Epochs | [1, 2, 3] | 2 |
| Batch size | 2 (with gradient accumulation 4) | effective 8 |
| Warmup steps | 10 | 10 |

### Cloud: Together AI

```bash
# Upload dataset
curl -X POST https://api.together.xyz/v1/files \
  -H "Authorization: Bearer $TOGETHER_API_KEY" \
  -F "file=@train.jsonl" -F "purpose=fine-tune"

# Launch fine-tuning
curl -X POST https://api.together.xyz/v1/fine-tunes \
  -H "Authorization: Bearer $TOGETHER_API_KEY" \
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct", "training_file": "file-xxxxx",
       "n_epochs": 3, "learning_rate": 1e-5, "suffix": "trading-v1"}'

# Result: model accessible as "alexglz/Qwen2.5-7B-Instruct-trading-v1"
```

---

## 7. Advanced Techniques

### DPO (Direct Preference Optimization) — Phase 2

After SFT, train on preference pairs (winning trades = "chosen", losing trades = "rejected"). Teaches the model not just "what's correct" but "what's better than the alternative." Requires a working SFT model first.

### Curriculum Learning — Recommended

Order training examples by difficulty:
1. **Easy** (first 30%): Strong trends, ADX > 35, clear breakouts, > 3% moves
2. **Medium** (next 40%): Moderate trends, mixed signals, 1.5-3% moves
3. **Hard** (final 30%): Weak trends, conflicting timeframes, borderline signals

Implementation: sort JSONL by absolute magnitude of price move.

### Synthetic Reasoning Augmentation — Consider

Use Claude/GPT-4 to generate higher-quality reasoning for existing labeled examples (~$5-10 for 1,000 examples). Only for the reasoning field — labels MUST come from real market data.

### NOT Recommended for This Project

- **RLHF with backtest rewards**: Too complex for 7-week timeline, sparse/delayed reward signal
- **Multi-task fine-tuning**: Mixing generation + evaluation dilutes focus on the primary task

---

## 8. Evaluation Framework

### Baseline Capture (Do FIRST)

Before training anything, run base Qwen 2.5 7B zero-shot on the test set and record all metrics. Without this, you can't prove improvement.

### Classification Metrics

| Metric | Target (fine-tuned) |
|---|---|
| Direction Accuracy | > 55% |
| Precision (per class) | > 50% |
| Recall (per class) | > 50% |
| F1 (macro) | > 0.50 |
| JSON Parse Success | > 95% |

### Backtest Metrics

| Metric | Target |
|---|---|
| Win Rate | > 50% |
| Profit Factor | > 1.2 |
| Avg Win / Avg Loss | > 1.5 |
| Sharpe Ratio | > 0.5 |
| Max Drawdown | Document |

See [Backtest Framework](./backtest-framework.md#metrics) for full metric definitions and simulation logic.

### Statistical Significance

- **McNemar's test** on classification (p < 0.05 = significant improvement)
- **Paired t-test** on per-trade P&L (baseline vs fine-tuned)

### Overfitting Detection

| Signal | Detection | Action |
|---|---|---|
| Train loss ↓, val loss ↑ | Plot both curves | Early stopping at val loss minimum |
| Perfect train accuracy, poor test | Compare metrics | Reduce epochs, increase dropout |
| Predictions cluster around training prices | Check distribution | Normalize prices (use % from entry) |
| Works on val but not test | Temporal regime change | Document as limitation |

### Deliverables

| Artifact | Format |
|---|---|
| Confusion matrix (base vs FT) | matplotlib heatmap |
| Classification report | Table (precision, recall, F1) |
| Equity curve | matplotlib line chart |
| Reliability diagram | matplotlib scatter |
| Training loss curve | matplotlib (from logs) |
| Per-pair accuracy breakdown | Bar chart |
| Statistical significance | p-value table |
| Error analysis (20 examples) | Qualitative table |

---

## 9. Implementation Timeline

| Phase | Duration | Deliverable |
|---|---|---|
| **0. Baseline capture** | 2 days | Zero-shot metrics on 200+ test examples |
| **1. Data pipeline** | 7 days | `generate_training_data.py` → `all_examples.jsonl` |
| **2. QLoRA training (local)** | 5 days | Fine-tuned weights + training curves |
| **3. Together AI training** | 3 days | Cloud model endpoint (parallel with Phase 2 eval) |
| **4. Evaluation** | 5 days | All metrics, charts, statistical tests |
| **5. Integration & demo** | 3 days | Model loaded in Ollama, A/B comparison |
| **6. Documentation** | 5 days | Technical report, presentation |

**Critical path**: ~21 working days (~4.5 weeks). Buffer: ~2.5 weeks within the 7-week timeline.

---

## 10. Key Decisions Summary

| Decision | Recommendation | Confidence |
|---|---|---|
| What to fine-tune | Generator node only | High |
| Method | QLoRA (4-bit) via Unsloth | High |
| Base model | Qwen 2.5 7B Instruct | High |
| Dataset size | 3,000-5,000 examples | High |
| Include NEUTRAL | **No** — discard ambiguous | High |
| Data format | Chat format matching `generator_node` exactly | Critical |
| Train/val/test split | Temporal (not random) | Critical |
| Labeling threshold | Dynamic (1.5× ATR) | High |
| Reasoning generation | Programmatic templates | High |
| Advanced technique | DPO as Phase 2 if time permits | Medium |
| Curriculum learning | Yes — sort by move magnitude | Medium |

> The single most impactful thing is **building a high-quality dataset with aggressive filtering**. A mediocre model on great data outperforms a great model on mediocre data. Spend 60% of effort on the data pipeline, 20% on training, 20% on evaluation.
