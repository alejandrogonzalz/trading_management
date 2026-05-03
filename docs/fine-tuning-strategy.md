# Fine-Tuning Strategy

**Technical plan for Approach 2: training an LLM on historical trading data.**

For context on why we're doing this, see [three-approaches.md](./three-approaches.md).
For the shared evaluation pipeline, see [backtest-framework.md](./backtest-framework.md).

---

## What Fine-Tuning Is

A general-purpose LLM (like Qwen 2.5 7B) understands language, math, and reasoning, but knows nothing about reading crypto indicators and producing trade setups. Fine-tuning feeds it thousands of historical examples: "given these indicators, here's what a good trade setup looks like."

After training, the model recognizes patterns it couldn't before — not because we changed the rules, but because we taught it what good analysis looks like in this specific domain.

**What changes**: The LLM's weights (its "brain"). Nothing else — same indicators, same pipeline, same output format.

**What doesn't change**: The scanner, indicator calculation, evaluator, risk management, LangGraph pipeline, or any other system component.

---

## Model Choice: Qwen 2.5 7B Instruct

| Factor | 7B | 14B |
|--------|-----|------|
| QLoRA VRAM | ~8 GB | ~16 GB |
| Training speed | 2× faster | Baseline |
| Inference speed | ~2s (Ollama) | ~4s |
| JSON output quality | Excellent | Slightly better |
| Overfitting risk | Lower | Higher (for our dataset size) |

The 7B model is the sweet spot: fits in GPU for both training and inference, fast enough for real-time use, and less prone to overfitting on ~12K examples.

---

## Training Method: QLoRA

Full fine-tuning of a 7B model requires 56+ GB VRAM — impractical. QLoRA (Quantized Low-Rank Adaptation) trains only ~0.5% of parameters while the base model stays frozen in 4-bit quantization.

- Base model: frozen, 4-bit quantized (~4 GB)
- LoRA adapters: trainable, ~35 MB
- Quality difference vs full fine-tuning: negligible for datasets under 50K examples

---

## Training Data

### Format

Each training example is a chat-format JSONL line matching the production prompt exactly:

```jsonl
{"messages":[
  {"role":"system","content":"You are a Senior Technical Analyst for a FUTURES trading system..."},
  {"role":"user","content":"Symbol: BTCUSDT\nIndicators: {\"1h\":{\"price\":67234.5,\"rsi\":62.3,\"macd_hist\":125.4,...}}"},
  {"role":"assistant","content":"{\"bias\":\"LONG\",\"entry\":67234.5,\"tp\":69100.0,\"sl\":66400.0,\"leverage\":5,\"reasoning\":\"Strong bullish breakout on 1h confirmed by 4h uptrend.\",\"quality\":\"HIGH\"}"}
]}
```

The system prompt, user format, and assistant JSON schema must match production exactly. Any mismatch degrades fine-tuning effectiveness.

### Dataset Stats

| Dimension | Value |
|-----------|-------|
| Total labeled samples | 17,353 |
| Training split | 12,147 (first 70% temporally) |
| Validation split | 2,603 (next 15%) |
| Test split | 2,603 (final 15%) |
| LONG : SHORT | 45.5% : 54.5% |
| Pairs represented | 20 |
| Timeframe | 1h |

### Temporal Split (Not Random)

```
Train:      months 1–4  (first 70%)
Validation: month 5     (next 15%)
Test:       month 6     (final 15%)
```

Random splits leak future information through autocorrelated market regimes. Temporal splits prevent this.

### The NEUTRAL Class Decision

**No NEUTRAL examples in training data.** Markets are neutral 60–70% of the time. A model that learns "when in doubt, say NEUTRAL" is useless. Instead, ambiguous candles are discarded during labeling. The model only trains on clear LONG and SHORT examples.

### Reasoning Generation

Reasoning strings are generated programmatically from the indicators — not written by hand. Templates combine relevant indicator readings into natural-language sentences:

```python
# Example: "RSI at 28 indicates oversold conditions, confirmed by bullish MACD crossover
# on 4h with ADX at 32 showing trend strength."
```

---

## Training Pipeline

### Local: QLoRA via Unsloth

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048, load_in_4bit=True,
)
model = FastLanguageModel.get_peft_model(
    model, r=16,
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha=32, lora_dropout=0.05,
)
# Train with SFTTrainer, export to GGUF, load in Ollama
```

### Cloud: Together AI

```bash
# Upload dataset
curl -X POST https://api.together.xyz/v1/files \
  -H "Authorization: Bearer $TOGETHER_API_KEY" \
  -F "file=@train.jsonl" -F "purpose=fine-tune"

# Launch fine-tuning
curl -X POST https://api.together.xyz/v1/fine-tunes \
  -H "Authorization: Bearer $TOGETHER_API_KEY" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct","training_file":"file-xxxxx",
       "n_epochs":3,"learning_rate":1e-5,"suffix":"trading-v1"}'
```

Result: model accessible as `alexglz/Qwen2.5-7B-Instruct-trading-v1` via the same OpenAI-compatible API.

### Hyperparameters

| Parameter | Range | Default |
|-----------|-------|---------|
| Learning rate | [1e-4, 2e-4, 5e-4] | 2e-4 |
| LoRA rank | [8, 16, 32] | 16 |
| LoRA alpha | 2× rank | 32 |
| Epochs | [1, 2, 3] | 2 |
| Batch size | 2 (gradient accumulation 4) | effective 8 |

---

## Evaluation Plan

### Target Metrics

| Metric | Zero-shot Baseline | Target (Fine-tuned) |
|--------|-------------------|---------------------|
| Direction Accuracy | 41.5% | > 55% |
| Win Rate | 11.5% | > 50% |
| Profit Factor | 1.91 | > 1.5 |
| F1 Score (macro) | ~0.40 | > 0.50 |
| JSON Parse Success | ~85% | > 95% |

### Statistical Significance

- **McNemar's test** on classification (p < 0.05 = significant improvement)
- **Paired t-test** on per-trade P&L (baseline vs fine-tuned on same samples)

### Overfitting Detection

| Signal | Detection | Action |
|--------|-----------|--------|
| Train loss ↓, val loss ↑ | Plot both curves | Early stopping at val loss minimum |
| Perfect train accuracy, poor test | Compare metrics | Reduce epochs, increase dropout |
| Predictions cluster around training prices | Check distribution | Normalize prices (use % from entry) |

---

## Advanced Techniques (If Time Permits)

**DPO (Direct Preference Optimization)**: After SFT, train on preference pairs — winning trades as "chosen," losing trades as "rejected." Teaches the model what's *better*, not just what's correct.

**Curriculum Learning**: Order training examples by difficulty — strong trends first, ambiguous setups last. Implementation: sort JSONL by absolute magnitude of price move.

---

## Related Docs

- [The 3 Approaches](./three-approaches.md) — Where fine-tuning fits in the bigger picture
- [Backtest Framework](./backtest-framework.md) — How the fine-tuned model will be evaluated
- [Action Plan](./action-plan.md) — Timeline for training execution
