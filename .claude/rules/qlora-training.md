# QLoRA Fine-Tuning — Running & Scaling

Fine-tunes Qwen 2.5 7B (4-bit) with Unsloth for trade-direction prediction.
Script: `langgraph/optimization/train_qlora.py`. Venv: `langgraph/.venv-finetuning/`.

## Run it (local, RTX 5070 Ti 16GB)

```powershell
cd C:\Users\alex\projects\trading_management\langgraph
# 1 epoch (recommended first pass), live output + saved log:
.\.venv-finetuning\Scripts\python.exe optimization\train_qlora.py --epochs 1 2>&1 | Tee-Object -FilePath logs\qlora_train.log

# Smoke test the loop (3 steps, tiny eval) before a long run:
.\.venv-finetuning\Scripts\python.exe optimization\train_qlora.py --max-steps 3 --max-eval 10
```

Outputs: result JSON → `optimization/results/qlora_optimization.json`,
adapters + GGUF → `backtest/data/models/qlora_qwen25_7b/`.

## Key constraints (learned the hard way)

- **`max_seq_length` must be ≥ 1024.** Every chat example is 877–933 tokens
  (prompt alone ~820). Unsloth's padding-free batching truncates `input_ids`
  but NOT `labels` when a sequence exceeds `max_seq_length`, crashing the fused
  cross-entropy loss with a batch-size mismatch on step 0. 1024 fits all samples
  → zero truncation. `_load_chat_dataset` also drops any example > `max_seq_length`
  as a safety net.
- **`batch_size=1`, `grad_accum=16`** keeps 1024-token training inside 16GB.
- **`per_device_eval_batch_size=1`** — the HF default of 8 OOMs at ~900 tokens.
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

### Instance options
| Instance | GPU | VRAM | ~$/hr | Notes |
|----------|-----|------|-------|-------|
| ml.g5.2xlarge | A10G | 24GB | ~$1.5 | cheapest workable, batch 8 |
| ml.g6e.xlarge | L40S | 48GB | ~$2.0 | best value, batch 16, FA2 |
| ml.p4d.24xlarge | A100 40GB | 40GB | ~$32 | overkill for one 7B QLoRA |

Recommended: **ml.g6e.xlarge** (L40S 48GB) → `batch_size=8`,
`gradient_accumulation_steps=2`, FA2 on. Expect ~1–2h/epoch, a few dollars total.

### Two ways to run
1. **SageMaker Notebook / Studio instance** — interactive, closest to the current
   workflow. Launch a GPU notebook, `git clone`, recreate the venv on Linux
   (`pip install unsloth` pulls a working FA2), bump `batch_size`, run the same
   `train_qlora.py`. Stop the instance when done so it stops billing.
2. **SageMaker Training Job** — managed/ephemeral. Push `dataset.jsonl` to S3,
   use the HuggingFace DLC (or a custom image with Unsloth) as the entry point,
   write artifacts back to S3. Spins down automatically; best for unattended runs.

### Changes needed vs local
- On Linux, raise `batch_size` to 8–16 and lower `gradient_accumulation_steps`
  to keep effective batch ~16–32 (`--batch-size 8`).
- Confirm FA2 is active in the startup banner (`FA2 = True`) — that's the speedup.
- `max_seq_length` stays 1024; the length/eval-batch constraints above still hold.

### Cost vs. local trade-off
Local is "free" but ties up the workstation for ~18h/epoch. SageMaker g6e is
~$2–5 for the whole run and frees the machine — worth it for multi-epoch sweeps
or the QLoRA config search in `optimization/configs/qlora.yaml`.
