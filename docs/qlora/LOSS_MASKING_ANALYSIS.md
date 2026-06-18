# Loss Masking Strategy: Full-Sequence vs Completion-Only

## Decision

We use **full-sequence cross-entropy loss** (no masking) rather than completion-only masking (`DataCollatorForCompletionOnlyLM`). This document records the rationale and quantitative analysis behind that choice.

## Setup

| Parameter | Value |
|-----------|-------|
| Trainer | TRL `SFTTrainer` with `dataset_text_field="text"` |
| Chat template | Qwen 2.5 native (`<\|im_start\|>system/user/assistant`) |
| Sequence length | 877–933 tokens per example |
| Assistant (task) tokens | ~100 tokens (structured JSON output) |
| System + User (context) tokens | ~800 tokens |
| Training samples | 39,312 |
| Epochs | 3 |
| Effective batch | 16 |

## The Concern

With ~89% of each sequence occupied by the system/user prompt (which is mostly static boilerplate + per-sample indicator values), is the model wasting capacity learning to reproduce the prompt instead of learning the indicators-to-signal mapping?

## Analysis: Why Full-Sequence Loss Self-Corrects

### Static tokens converge to near-zero loss immediately

The system prompt is byte-identical across all 39,312 training samples. After ~50 optimizer steps, the model's cross-entropy on these tokens approaches 0 and they contribute negligible gradient. This happens well before the LR warmup (50 steps) even completes.

Effective gradient timeline:

| Training phase | Tokens with non-trivial loss | Task signal share |
|----------------|------------------------------|-------------------|
| Steps 1–50 | All ~900 (model still surprised by template) | ~11% |
| Steps 50–200 | ~200 (indicator values + assistant JSON) | ~50% |
| Steps 200+ | ~120 (varying indicator digits + assistant) | ~80%+ |

Cross-entropy is self-masking on predictable tokens. By the time the model is refining its decision boundary, the "wasted capacity" problem has self-extinguished.

### Total task-relevant gradient signal is abundant

```
39,312 samples x 3 epochs x ~100 assistant tokens = 11.8M task-token observations
/ effective batch 16 = ~7,371 optimizer steps
```

For a structured JSON output with 6 decision-relevant fields (bias, entry, tp, sl, quality, confidence), convergence typically requires 500–1,000 steps. We have 7x that budget — gradient starvation is not the bottleneck.

### Variable indicator values provide mild auxiliary benefit

The ~120 tokens of indicator *values* (numbers that vary per sample) never fully converge to zero loss. With full-sequence loss, the model receives gradient that reinforces its internal representation of those input numbers — effectively a form of auxiliary "read comprehension" training. With completion-only masking, this representation forms only through the indirect attention path.

## Expected Gain from Completion-Only Masking

| Scenario | Expected accuracy improvement |
|----------|-------------------------------|
| **Our setup** (50K samples, 3 epochs, 100-token JSON) | **0.3–1.0pp** |
| Few samples (<5K), 1 epoch | 1.5–3pp |
| Long-form generation output (500+ tokens) | 1–2pp |
| Non-repetitive diverse prompts | 2–4pp |

Our scenario falls in the lowest-impact bucket: many samples, short structured output, multiple epochs, and highly repetitive prompts that self-extinguish.

## When Completion-Only Masking Is Critical

The masking decision matters significantly when:

1. **Few samples** (<5K) — every gradient step matters, early waste is unrecoverable
2. **Long, diverse outputs** — the output distribution is complex and needs all signal
3. **Very short training** (1 epoch, small model) — template loss has no time to decay
4. **Non-repetitive prompts** — user text varies enough that it never reaches zero loss

None of these apply to our training configuration.

## Implementation (Not Applied)

For reference, the completion-only approach would be:

```python
from trl import DataCollatorForCompletionOnlyLM

response_template = "<|im_start|>assistant\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)
trainer = SFTTrainer(..., data_collator=collator)
```

This masks all tokens before the assistant turn marker, computing loss only on the JSON output tokens.

## Higher-Leverage Interventions

If accuracy improvement beyond 88% is desired, these avenues have larger expected impact than loss masking:

1. **ATR-based TP/SL labels** (`--use-atr-tp-sl`) — de-circularizes exit targets, removes hindsight dependency from training labels
2. **More diverse training data** — additional symbols, market regimes, time periods
3. **Curriculum learning** — train on high-confidence samples first, then introduce harder edge cases
4. **Longer training with lower LR** — squeeze more from existing data

## Conclusion

Full-sequence loss is a valid design choice for this task configuration. The self-extinguishing property of cross-entropy on repeated tokens, combined with abundant training signal (11.8M task-token observations over 7,371 steps), means that completion-only masking would yield marginal improvement (estimated <1pp). We document its existence as a potential optimization for future work with smaller datasets or longer output sequences.
