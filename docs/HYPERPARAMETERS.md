# QLoRA Hyperparameters — What They Are and Why They Matter

A precise explanation of every hyperparameter used in `train_qlora.py`, organized by
function. Each section answers three questions: what does it control, what value did
we choose and why, and what goes wrong if you pick the wrong value.

---

## Background — What a Hyperparameter Is Here

When you fine-tune a model, there are two kinds of numbers:

- **Weights** (parameters): the numbers inside the model that change during training.
  For QLoRA that means the LoRA adapter matrices A and B — ~40 million numbers that
  the optimizer updates on every step.
- **Hyperparameters**: the settings you choose before training starts that control
  *how* the weights are updated. They are not learned — you set them, and the training
  outcome depends heavily on getting them right.

The hyperparameters in this project fall into four groups:

| Group | What it controls | Params |
|-------|-----------------|--------|
| **LoRA architecture** | Shape and scope of the adapters | `rank`, `alpha`, target modules |
| **Learning dynamics** | How fast and how long we update weights | `lr`, LR schedule, `warmup_steps`, `epochs` |
| **Memory management** | Fitting 1024-token samples on the GPU | `batch_size`, `grad_accum`, `max_seq_length` |
| **Regularization** | Preventing the model from memorizing training data | `weight_decay`, early stopping, `seed` |

---

## Group 1 — LoRA Architecture

### Rank (`rank = 16`)

**What it controls:** the size of the adapter matrices.

Each frozen weight matrix W in the transformer has shape `[d_out × d_in]` —
a large matrix. Instead of learning a full correction to W, LoRA learns two small
matrices:

```
A: [d_in  × rank]     e.g.  [4096 × 16]
B: [rank  × d_out]    e.g.  [16 × 4096]

Correction: ΔW = A × B   shape: [4096 × 4096]  but only 2 × 4096×16 = 131,072 params
Full correction would need: 4096 × 4096 = 16,777,216 params
```

Rank = 16 means the adapter can only represent corrections that lie in a 16-dimensional
subspace of the full matrix. This is the "bottleneck" that makes LoRA parameter-efficient.

**Why 16?**

```
rank =  4 → ~10M trainable params → underfits; misses subtler indicator relationships
rank = 16 → ~40M trainable params → fits this task; enough capacity for JSON output
rank = 64 → ~160M trainable params → more VRAM, longer training, marginal gain for 39K samples
```

For a narrow, well-defined task (indicators → JSON) with 39K examples, rank 16 is the
established sweet spot. Higher ranks help when the task has broad, diverse structure
(e.g. general instruction following).

**Sensitivity: LOW-MEDIUM.** Changing from 8 to 32 rarely moves accuracy by more than
1–2%. Pick it once and leave it.

---

### Alpha (`alpha = 32`)

**What it controls:** how much weight the adapter output gets relative to the frozen base.

The adapter's contribution is scaled before being added to the base model output:

```
Effective correction: ΔW = (alpha / rank) × A × B

With alpha=32, rank=16:  multiplier = 32/16 = 2.0

At inference: output = (W + 2.0 × A×B) × input
```

A multiplier of 2× means the adapter "speaks twice as loud" as a rank-16 adapter
without scaling would. This compensates for the fact that small rank means small
magnitude in A×B.

**Why 32?** The rule of thumb `alpha = 2 × rank` is the standard default. It keeps the
adapter's contribution in a reasonable range without requiring a separate LR search.

**Sensitivity: LOW.** As long as you use `alpha = 2 × rank`, this value has negligible
impact. Changing it shifts the effective learning rate slightly — which is easier to
control through `lr` directly.

---

### Target modules

**What it controls:** which weight matrices in the transformer get an adapter injected.

This project targets all projection matrices in attention and all feed-forward matrices:

```
Attention (per layer):     q_proj, k_proj, v_proj, o_proj   (4 matrices)
MLP / feed-forward:        gate_proj, up_proj, down_proj      (3 matrices)
                                                              ──────────
Total per layer:           7 matrices
Total (7 × 28 layers):     196 adapter pairs
```

**Why these modules?**
- `q_proj`, `k_proj`, `v_proj`: control which tokens attend to which — adapting these
  teaches the model *what to pay attention to* (e.g. the RSI value in the user prompt).
- `o_proj`: combines attended information back into the residual stream.
- `gate_proj`, `up_proj`, `down_proj`: the MLP that transforms attended information —
  adapting these teaches the model *what to do with* the attended signals (e.g. RSI>70
  → overbought → SHORT bias).

Targeting more modules increases trainable params and expressiveness; targeting fewer
reduces VRAM and may underfit.

---

## Group 2 — Learning Dynamics

### Learning rate (`lr = 2e-5`)

**What it controls:** how large a step the optimizer takes on each weight update.

After computing the gradient (the direction that reduces loss), the optimizer updates
each adapter weight by:

```
new_weight = old_weight − lr × gradient_direction
```

A high LR takes big steps — fast learning but risks overshooting the minimum and
diverging. A low LR takes tiny steps — stable but may not converge in 3 epochs.

```
lr = 1e-3  → steps too large; loss spikes and diverges within the first 100 steps
lr = 2e-5  → chosen value; fast early learning that stabilizes as LR decays
lr = 1e-6  → steps too small; model barely moves in 3 epochs; underfits
```

**Why 2e-5?** This is the most common and well-validated LR for QLoRA fine-tuning on
instruction-tuned models (Qwen, Llama, Mistral families). Lower values like 1e-5 are
sometimes used for larger ranks or longer runs where stability matters more.

**Sensitivity: HIGH — the single most impactful hyperparameter.** If you only tune
one thing, tune this.

---

### LR schedule (cosine decay)

**What it controls:** how the learning rate changes over the course of training.

The LR does not stay at 2e-5 the whole time. It follows two phases:

```
Phase 1 — Warmup (steps 0 → 50):
  LR ramps linearly from 0 to 2e-5
  Prevents instability when the optimizer has no gradient statistics yet

Phase 2 — Cosine decay (steps 50 → end):
  LR decays following a cosine curve from 2e-5 to ~0

  2e-5 ─┐
        │╲
        │ ╲
        │  ╲___
        │      ╲____
        │           ╲_________ ~0
        └─────────────────────►
        50              4,904 steps
```

The cosine shape means the LR drops quickly at first (when big improvements are still
happening), then very slowly near the end (when the model is already close to converged
and large updates would destabilize it).

---

### Warmup steps (`warmup_steps = 50`)

**What it controls:** how many steps before the LR reaches its peak value.

At step 0, the Adam optimizer has no memory of past gradients (its internal moment
estimates are zero). The first few gradient updates are therefore noisy estimates. If
the LR is already at 2e-5, those noisy updates can push the adapter weights in
bad directions from the start.

Warmup fixes this by starting at LR=0 and slowly increasing it over 50 steps. By step
50, Adam has seen enough gradients to build reliable estimates, and the full LR is safe.

**Why 50?** ~2% of steps per epoch. Enough for Adam to initialize; not so long that
it delays meaningful learning. Standard value across QLoRA literature.

---

### Epochs (`epochs = 3`)

**What it controls:** how many times the model sees the full training set.

One epoch = one complete pass over all 39,313 training samples.

```
Epoch 1:
  Steps 1–250:     Loss drops fast — model learns easy patterns
                   (correct JSON schema, RSI > 70 = overbought, ATR-sized SL)
  Steps 250–2452:  Loss drops slower — subtler patterns
                   (multi-TF alignment, MACD + ADX combinations)
  End of epoch:    Evaluate on val set → save checkpoint if best

Epoch 2:
  Same samples, shuffled differently. Loss refines further.
  Model learns to be consistent, not just roughly right.

Epoch 3:
  Diminishing returns. Overfitting risk appears.
  If val loss stops improving → EarlyStoppingCallback stops training here.
```

**Why 3?** At 39K samples, 1 epoch gives a good first model but misses edge cases.
3 epochs gives enough repetition to solidify patterns without memorizing the training
data. 5+ epochs typically starts overfitting at this dataset size.

The actual training may stop earlier: `EarlyStoppingCallback(patience=3)` halts if
`eval_loss` fails to improve for 3 consecutive evaluations (every 250 steps).

---

## Group 3 — Memory Management

### Batch size (`batch_size = 2`)

**What it controls:** how many training samples the GPU processes simultaneously in one
forward+backward pass.

Each 1024-token sample requires memory for:
- Input activations across all 28 transformer layers
- Gradient tensors for each A and B adapter matrix
- Adam optimizer state (two running averages per weight = 2× the weight memory)

At 1024 tokens and bf16 precision, one sample uses roughly 2–4 GB of VRAM during the
backward pass. That limits how many samples can run in parallel:

```
GPU VRAM  │  max batch_size  │  why
──────────┼──────────────────┼────────────────────────────────
16 GB     │  1               │  RTX 5070 Ti local
48 GB     │  2–4             │  L40S on SageMaker (4+ OOMs)
80 GB     │  8               │  A100 on RunPod
```

Larger batch size = more accurate gradient estimate per step, but more VRAM.

**Why 2 on cloud (L40S 48GB)?** batch=4 OOMed on the backward pass during our
SageMaker run. batch=2 is the confirmed safe ceiling on L40S for 1024-token samples.

---

### Gradient accumulation (`grad_accum = 8`)

**What it controls:** how many forward passes are summed before a weight update.

With `batch_size=2` and `grad_accum=8`, the training loop works like this:

```
Step 1: forward + backward on batch 1 (2 samples) → accumulate gradients, do NOT update
Step 2: forward + backward on batch 2 (2 samples) → accumulate gradients, do NOT update
...
Step 8: forward + backward on batch 8 (2 samples) → NOW update weights

One weight update sees: 2 × 8 = 16 effective samples
```

This is mathematically equivalent to training with `batch_size=16` in one shot — the
gradients average out the same way — but uses only the memory of `batch_size=2`.

**Effective batch = batch_size × grad_accum = 2 × 8 = 16.** This is the number that
matters for training stability.

```
effective_batch < 8   → gradients too noisy; unstable loss
effective_batch = 16  → standard for QLoRA; smooth, stable training
effective_batch > 64  → each epoch has very few updates; slow to converge
```

The trade-off: higher `grad_accum` means fewer weight updates per epoch (2,452 / 8 vs
2,452 / 1), so the model learns more slowly per wall-clock minute.

---

### max_seq_length (`max_seq_length = 1024`)

**What it controls:** the maximum number of tokens in one training example. Any example
longer than this limit is either truncated or dropped.

Our chat examples are 877–933 tokens:
```
System prompt:    ~250 tokens   (trading analyst persona + JSON schema)
User prompt:      ~570 tokens   (symbol name + 3 TFs × 9 indicators)
Assistant answer: ~100 tokens   (JSON: bias, entry, tp, sl, reasoning)
──────────────────────────────────────────────────────────────
Total:            ~877–933 tokens
```

1024 gives 91–147 tokens of headroom. This is important because Unsloth uses
**padding-free batching** (samples are packed together to maximize GPU utilization).
If any sample is truncated, `input_ids` is shortened but `labels` is not, causing a
shape mismatch that crashes the fused cross-entropy loss at step 0.

**This is a hard constraint, not a tunable parameter.** Do not lower it below 1024.

---

## Group 4 — Regularization

### Weight decay (`weight_decay = 0.01`)

**What it controls:** L2 regularization applied to the adapter weights.

On each weight update, in addition to moving in the gradient direction, we also shrink
the weight slightly toward zero:

```
new_weight = old_weight × (1 − lr × weight_decay) − lr × gradient
           = old_weight × 0.9999998 − lr × gradient   (at lr=2e-5, wd=0.01)
```

This prevents any single adapter element from growing very large and dominating the
output — a form of overfitting where one pattern is weighted too heavily.

0.01 is mild: the shrinkage per step is tiny (0.00002%), so it does not impede
learning but provides a floor against extreme weight values.

**Sensitivity: NEGLIGIBLE.** Standard value for LLM fine-tuning; leave it at 0.01.

---

### Early stopping (`patience = 3`)

**What it controls:** when to stop training if the model stops improving on the
validation set.

Every 250 steps, `train_qlora.py` evaluates the model on `val.jsonl` and records
`eval_loss`. If `eval_loss` has not improved (decreased) for 3 consecutive evaluations
(750 steps), training stops automatically.

```
Evaluation log example:
  Step  250: eval_loss = 0.921  ← new best → save checkpoint
  Step  500: eval_loss = 0.874  ← new best → save checkpoint
  Step  750: eval_loss = 0.861  ← new best → save checkpoint
  Step 1000: eval_loss = 0.863  ← no improvement (round 1)
  Step 1250: eval_loss = 0.865  ← no improvement (round 2)
  Step 1500: eval_loss = 0.869  ← no improvement (round 3) → STOP
```

`load_best_model_at_end=True` then restores the checkpoint from step 750, discarding
the three steps of mild overfitting that followed.

**Why patience=3?** One evaluation could be noise. Three consecutive non-improvements
is a strong signal that the model has stopped learning from the training data.

---

### Seed (`seed = 42`)

**What it controls:** the starting point for all random number generators in the run.

Three sources of randomness are seeded:
- **LoRA adapter initialization**: the initial values of A and B matrices (B is
  initialized to all zeros; A is initialized from a normal distribution — the seed
  controls that distribution)
- **Training data shuffle order**: the order in which the 39,313 samples are presented
  across epochs
- **Dropout** (if any dropout is applied inside the adapters)

Setting `seed=42` makes all three deterministic: the same seed on the same hardware
produces identical model weights at every step, making experiments reproducible.

---

## How the Hyperparameters Interact

```
rank, alpha, target modules
    │
    └──► Shape of adapters (how many params, how expressive)
              │
              └──► Memory footprint per sample
                        │
                        └──► max batch_size the GPU can hold
                                  │
                                  └──► grad_accum needed to reach effective_batch=16
                                            │
                                            └──► steps per epoch = 39313 / (batch × grad_accum)
                                                      │
                                                      └──► total steps = steps_per_epoch × epochs
                                                                │
                                                                └──► LR schedule shape and decay rate
```

The key cascade: **rank** determines adapter size → adapter size determines peak VRAM
→ peak VRAM constrains `batch_size` → `batch_size` + `grad_accum` = effective batch →
effective batch determines how many weight updates happen per epoch → weight updates
interact with the LR schedule to determine how fast the model converges.

---

## Reading the Training Logs

Every 50 steps, the trainer logs a line:

```
{'loss': 0.7813, 'grad_norm': 0.148, 'learning_rate': 1.42e-05, 'epoch': 0.36}
```

| Field | What it measures | Healthy range |
|-------|-----------------|---------------|
| `loss` | Cross-entropy on the current batch's assistant tokens | Starts ~1.8, should drop to ~0.4–0.8 |
| `grad_norm` | Euclidean norm of all adapter gradients combined | 0.1–3.0; >10 = unstable training |
| `learning_rate` | Current LR after warmup + cosine decay | 2e-5 → ~0 over the full run |
| `epoch` | Fraction of total training complete | 0.0 → number of epochs |

And every 250 steps (checkpoint save):

```
{'eval_loss': 0.8134, 'eval_runtime': 42.1, ...}
```

| Field | What it means |
|-------|--------------|
| `eval_loss` | Cross-entropy on val.jsonl — the key metric for early stopping |
| Decreasing `eval_loss` | Model is generalizing — the adapter changes are making it better on unseen val data |
| `eval_loss` rising while `loss` falls | Overfitting — the model is memorizing training samples |

The script also emits a summary at the end of training:

```
overfitting_diagnostics:
  train_accuracy: 0.847
  val_accuracy:   0.724
  test_accuracy:  0.701
  train_val_gap:  0.123   ← gap > 0.15 suggests overfitting
  baseline:       0.510   ← heuristic majority-class baseline
```

A gap below 0.10 between train and val/test accuracy is healthy. A gap above 0.15
warrants skepticism — the model may be learning training-set-specific patterns rather
than generalizable ones.

---

## This Project's Values — Summary

| Hyperparameter | Value | Primary constraint |
|---------------|-------|-------------------|
| `lora_rank` | 16 | Enough capacity for a single narrow task |
| `lora_alpha` | 32 | Standard `2 × rank` scaling |
| Target modules | q/k/v/o + gate/up/down | Cover attention + MLP transforms |
| `lr` | 2e-5 | QLoRA sweet spot for instruction-tuned models |
| LR schedule | cosine + 50-step warmup | Standard; prevents early instability |
| `epochs` | 2–3 | 39K samples; early stopping guards against overfitting |
| `batch_size` | 2 (cloud) / 1 (local) | L40S OOMs at batch=4; RTX at batch=2 |
| `grad_accum` | 8 (cloud) / 16 (local) | Targets effective batch = 16 |
| `max_seq_length` | 1024 | Hard floor — samples are 877–933 tokens |
| `weight_decay` | 0.01 | Mild regularization; standard default |
| Early stopping patience | 3 | 3 consecutive eval steps without improvement |
| `seed` | 42 | Reproducibility |

---

## Files Referenced

| File | Role |
|------|------|
| `optimization/qlora/train_qlora.py` | Script where all hyperparameters are applied |
| `optimization/qlora/run_cloud.sh` | Cloud run wrapper — sets cloud hyperparameter values |
| `optimization/qlora/run_local.sh` | Local run wrapper — sets local hyperparameter values |
| `docs/QLORA_FINETUNING.md` | Status, config decisions, pipeline diagrams, runbook |
| `docs/DATA_PIPELINE.md` | How the training data (the input to fine-tuning) is built |
