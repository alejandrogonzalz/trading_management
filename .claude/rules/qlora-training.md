# QLoRA Fine-Tuning — Concepts, Hyperparameters & Running

Fine-tunes Qwen 2.5 7B (4-bit) with Unsloth for trade-direction prediction.
Script: `langgraph/optimization/qlora/train_qlora.py`. Venv: `langgraph/.venv-finetuning/`.

> **Status & roadmap** (where we are, the SageMaker plan, what's left) lives in
> [`docs/QLORA_FINETUNING.md` §0](../../docs/QLORA_FINETUNING.md). This file is the
> commands/constraints reference.

---

## How fine-tuning works (conceptual)

### The core idea

Qwen 2.5 7B already "knows" language, reasoning, and JSON formatting. But it has
never seen our task: given multi-TF crypto indicators → output a trade setup JSON.
Fine-tuning = showing it 39,312 correct examples until it learns the mapping.

### What we're training (LoRA adapters)

The model has 7 billion parameters. We **freeze all of them** and inject tiny
"adapter" matrices into each transformer layer:

```
Original layer (frozen):     W = [7B params, untouched]
LoRA adapter (trainable):    ΔW = A × B   where A is [dim × rank], B is [rank × dim]

Effective weight at inference: W + ΔW
```

Target modules in the script (`load_model()`):
- Attention: `q_proj`, `k_proj`, `v_proj`, `o_proj`
- MLP: `gate_proj`, `up_proj`, `down_proj`

That's 7 matrices per layer × 28 layers = **196 adapter pairs**.
Total trainable: ~40M params (0.6% of 7B).

### What one training step does

```
1. Take a training example:
   System: "You are a Senior Technical Analyst..."
   User:   "Symbol: BTCUSDT\nIndicators: {RSI: 42, ADX: 31, EMA aligned...}"
   Assistant: {"bias": "LONG", "entry": 67500, "tp": 69200, "sl": 66800, ...}

2. Feed the ENTIRE conversation to the model

3. Model predicts the next token at every position, but loss is ONLY computed
   on the assistant's tokens (the model must learn to generate the correct answer)

4. Loss = cross-entropy (how wrong was the prediction vs the actual tokens?)

5. Backpropagate gradient → update ONLY the LoRA adapter matrices (A, B)
   The 7B base weights never change

6. Repeat × ~2,457 steps/epoch × 3 epochs = ~7,371 total weight updates
```

### Why 4-bit quantization (the "Q" in QLoRA)

The frozen base weights are stored in 4 bits instead of 16:
- Normal: 7B × 16 bits = ~14 GB VRAM (just the model)
- 4-bit:  7B × 4 bits  = ~3.5 GB VRAM (fits on consumer GPUs)
- LoRA adapters still compute in bf16 for gradient accuracy

### What the model "learns"

| Before (zero-shot) | After (fine-tuned) |
|---------------------|-------------------|
| Inconsistent JSON formatting | Always valid `{"bias", "entry", "tp", "sl", "reasoning"}` |
| Generic stop-loss guesses | SL at entry ± 1×ATR (learned from labels) |
| Can't combine multi-TF signals | Multi-TF alignment drives bias decisions |
| Sometimes hallucinates | Reasoning matches indicator patterns it's seen |

### The training loop across epochs

```
Epoch 1: Model sees all 39,312 samples once
  - Steps 1-250:    Loss drops fast (easy patterns: RSI>70 → overbought)
  - Steps 250-1000: Subtler patterns (multi-TF alignment, ATR sizing)
  - Step 2457:      End of epoch. Eval on val set → save if best.

Epoch 2: Same samples again, different order
  - Refines learned patterns. Loss drops further.

Epoch 3: Final pass
  - Diminishing returns. Overfitting risk starts.
  - We track eval_loss and keep the BEST checkpoint, not the last.
```

---

## Hyperparameters explained

### Learning rate (`lr=2e-5`)

How much each gradient update changes the adapter weights.
- Too high (1e-3): weights oscillate, loss spikes, model diverges
- Too low (1e-6): barely learns in 3 epochs, wastes compute
- 2e-5 is the sweet spot for QLoRA on instruction-tuned models

Combined with **cosine schedule**: LR starts at 2e-5, decays smoothly to ~0 by
the end of training. Prevents large updates in late training when the model is
already good.

### LoRA rank (`rank=16`)

The bottleneck dimension of the adapter matrices (A is `[dim × rank]`, B is `[rank × dim]`).
- rank=4: very few trainable params, underfits complex tasks
- rank=16: good balance — 40M params, enough capacity for our JSON task
- rank=64: more capacity but more VRAM, slower, risk of overfitting on 39K samples

Think of it as: "how many independent directions can the adapter adjust the
model's behavior in?" 16 is plenty for a single well-defined task.

### LoRA alpha (`alpha=32`)

Scaling factor applied to the adapter output: `ΔW = (alpha/rank) × A × B`.
With alpha=32 and rank=16, the multiplier is 2×. This means the adapter's
contribution is amplified relative to the frozen weights.

Rule of thumb: `alpha = 2 × rank` is a safe default.

### Epochs (`epochs=3`)

How many times the model sees the full training set.
- 1 epoch: learns the basics, might underfit edge cases
- 3 epochs: good for 39K samples — enough repetition without memorizing
- 5+ epochs: risk of overfitting (model memorizes training data, fails on test)

We mitigate overfitting two ways: `load_best_model_at_end=True` keeps the best
checkpoint (lowest `eval_loss`), and a real `EarlyStoppingCallback(patience=3)`
stops once `eval_loss` stops improving — so a 3-epoch run may end early. `evaluate()`
also reports the **train−test accuracy gap**, a heuristic **baseline**, and a
persisted **loss curve** so overfitting can actually be judged (not just assumed).

### Batch size & gradient accumulation

```
effective_batch = batch_size × gradient_accumulation_steps
```

| Setting | batch_size | grad_accum | effective | Where |
|---------|-----------|------------|-----------|-------|
| Local (16GB) | 1 | 16 | 16 | RTX 5070 Ti |
| SageMaker (48GB) | 8 | 16 | 128 | ml.g6e.xlarge |

- **batch_size**: how many samples the GPU processes in parallel per step.
  Limited by VRAM (each 1024-token sample needs ~2-4GB of activations at batch=1).
- **gradient_accumulation**: accumulates gradients over N mini-batches before
  updating weights. Gets the benefits of a large batch without the VRAM cost.
- **effective batch**: larger = smoother gradients, more stable training, but
  each epoch has fewer weight updates (2,457 steps at batch=16 vs ~307 at batch=128).

### Warmup steps (`warmup=50`)

LR ramps linearly from 0 → 2e-5 over the first 50 steps. Prevents early
instability when the model hasn't calibrated its gradients yet. After warmup,
the cosine decay takes over.

### Weight decay (`weight_decay=0.01`)

L2 regularization — penalizes large adapter weights slightly. Prevents any
single adapter from dominating. 0.01 is mild and standard for LLM fine-tuning.

### max_seq_length (`max_seq_length=1024`)

Maximum token length per training example. Our samples are 877-933 tokens.
1024 fits everything with headroom. Lowering this truncates examples and causes
a crash (Unsloth padding-free batching bug — see constraints below).

### Seed (`seed=42`)

Deterministic initialization for LoRA adapters, data shuffling, and dropout.
Ensures reproducible results across runs.

---

## Run it (local, RTX 5070 Ti 16GB)

```powershell
cd C:\Users\alex\projects\trading_management\langgraph
# 1 epoch (recommended first pass), live output + saved log:
.\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py --epochs 1 2>&1 | Tee-Object -FilePath logs\qlora_train.log

# Smoke test the loop (3 steps, tiny eval) before a long run:
.\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py --max-steps 3 --max-eval 10

# Resume an interrupted run from the latest checkpoint-N in output_dir:
.\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py --epochs 3 --resume 2>&1 | Tee-Object -FilePath logs\qlora_train.log
```

**Crash recovery:** checkpoints save every 250 steps (`save_total_limit=3`). If a
run dies mid-way, just re-launch with the same args plus `--resume` — it picks up
from the last `checkpoint-N` instead of restarting at step 0. Without `--resume`
the run starts fresh. (Note: a leftover smoke-test `checkpoint-3` in `output_dir`
counts as a checkpoint — delete it before a real `--resume` if you don't want to
resume from it.)

Outputs: result JSON → `optimization/qlora/results/qlora_<tag>.json` (+ canonical
`qlora_optimization.json`), loss curve → `results/<tag>_loss_curve.{json,png}`,
adapters + GGUF → `backtest/data/models/<tag>/`. Pass `--tag` to name a run.

> **For real runs use the wrappers, not a sweep:** `run_cloud.sh` (one SageMaker
> model on the fixed split, full test, no `--max-eval`) or `run_local.sh` (one local
> model). `dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles` first
> or financial metrics come out 0. The old 3-/5-config sweep + its 92% result are
> invalid (leakage) and archived — see `docs/AUDITORIA_QLORA_LEAKAGE_OVERFITTING.md`.

## Key constraints (learned the hard way)

- **`max_seq_length` must be ≥ 1024.** Every chat example is 877–933 tokens
  (prompt alone ~820). Unsloth's padding-free batching truncates `input_ids`
  but NOT `labels` when a sequence exceeds `max_seq_length`, crashing the fused
  cross-entropy loss with a batch-size mismatch on step 0. 1024 fits all samples
  → zero truncation. `_load_chat_dataset` also drops any example > `max_seq_length`
  as a safety net.
- **`batch_size=1`, `grad_accum=16`** keeps 1024-token training inside 16GB.
- **`per_device_eval_batch_size=1`** — the HF default of 8 OOMs at ~900 tokens.
- **Never cut the split by file position.** `_temporal_split` sorts globally by
  `timestamp` + embargo; `dataset.jsonl` is grouped by symbol, so a positional cut
  is a per-symbol split that leaks market regime. Keep eval on the FULL test (drop
  `--max-eval`) so it spans multiple symbols, not just LINK.
- Attention runs on the slow eager path locally: `FA [Xformers = None. FA2 = False]`.
  FlashAttention-2 / xformers are painful to install on Windows. This is the
  main throughput bottleneck.

## Local timing

~28s/step (variable 9–65s with padding-free packing), ~2,457 steps/epoch →
**~15–19h per epoch**. Per-step time fluctuates; judge ETA from the cumulative
average, not tqdm's instantaneous estimate. The first run compiles + caches
Triton kernels to disk (`unsloth_compiled_cache/`), so later runs start faster.

## Faster: AWS SageMaker (3–10× quicker)

The local box is the bottleneck, not the code. A cloud Linux GPU is much faster
because (a) more VRAM allows `batch_size` 8–16 instead of 1, and (b)
FlashAttention-2 installs cleanly on Linux. Both together typically cut a
1-epoch run from ~18h to **~1–3h**.

The dataset is tiny (`backtest/data/labeled/dataset.jsonl`, ~56k samples) so
data transfer is trivial.

Full step-by-step + cost analysis: see [`optimization/qlora/EC2_GUIDE.md`](../../langgraph/optimization/qlora/EC2_GUIDE.md).

### Instance options
| Instance | GPU | VRAM | ~$/hr | Notes |
|----------|-----|------|-------|-------|
| ml.g5.xlarge | A10G | 24GB | ~$1.4 | cheapest workable, batch 4-8 |
| **ml.g6e.xlarge** | **L40S** | **48GB** | **~$2.0** | **recommended**, batch 8-16, FA2 |
| ml.p4d.24xlarge | A100 40GB | 40GB | ~$32 | overkill for one 7B QLoRA |

Recommended: **ml.g6e.xlarge** (L40S 48GB) → `batch_size=8`,
`gradient_accumulation_steps=16`, FA2 on. Expect ~1–2h/epoch, ~$6-12 for 3 epochs.

### Changes needed vs local
- Raise `batch_size`: **2** without FA2 (batch 4 OOMs on the backward pass — the
  learned SageMaker ceiling), **try 4** once FA2 is confirmed (FA2 cuts peak VRAM).
  `run_cloud.sh` uses `--batch-size 2 --grad-accum 8`; bump it only after a
  `--max-steps 3` smoke test passes at the higher batch.
- Confirm FA2 is active in the startup banner (`FA2 = True`).
- `max_seq_length` stays 1024; the length/eval-batch constraints still hold.
- **No tmux on SageMaker Studio** — launch with `nohup ... &` so the job survives
  browser disconnects (note the PID, follow `tail -f .../run_cloud.log`). On a raw
  EC2 box tmux works too, but nohup is the portable default.

### Cost vs. local trade-off
Local is "free" but ties up the workstation for ~2 days (3 epochs). SageMaker g6e is
~$6–12 for the whole run and frees the machine — worth it for multi-epoch training.
