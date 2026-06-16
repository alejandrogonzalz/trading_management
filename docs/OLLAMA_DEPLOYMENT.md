# Running the Fine-Tuned Model Locally (Ollama)

This guide covers deploying the QLoRA fine-tuned Qwen 2.5 7B model for **inference only** —
not training. Inference requirements are much lower than training.

---

## Minimum Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| RAM | 8 GB | 16 GB |
| VRAM (GPU) | 0 GB (CPU-only works) | 6 GB+ |
| Disk | 6 GB free | 10 GB free |
| OS | Windows 10 / macOS 12 / Linux | any |

### Speed expectations by hardware

| Hardware | VRAM | Speed | Notes |
|----------|------|-------|-------|
| CPU only (no GPU) | — | ~2–5 tok/s | Works, but slow. Usable for testing. |
| RTX 3060 / RX 6600 | 8 GB | ~20–30 tok/s | Good. Full GPU offload. |
| RTX 3070 / RX 6700 XT | 8 GB | ~30–40 tok/s | Great. |
| RTX 4060 Ti / RTX 5070 Ti | 16 GB | ~50–80 tok/s | Excellent. Full offload with headroom. |
| A100 / H100 (cloud) | 40–80 GB | ~100+ tok/s | Overkill for inference. |

The model is exported as **Q4_K_M GGUF** (4-bit quantized, 4.4 GB).
Any machine with 8 GB of system RAM can run it — GPU just makes it faster.

---

## Prerequisites

1. **Ollama installed** → https://ollama.com/download  
   Verify: `ollama --version`

2. **DVC installed** (to pull the model from S3) → `pip install dvc[s3]`  
   Verify: `dvc --version`

3. **AWS credentials configured** (to pull from `s3://trading-management-dvc/`)

   You need an AWS IAM user with **read access to the S3 bucket**.
   Ask the repo owner (Alex) for an Access Key ID + Secret Access Key.

   Once you have them, configure the AWS CLI:

   ```bash
   # Install AWS CLI if needed: https://aws.amazon.com/cli/
   aws configure
   ```

   It will prompt for:
   ```
   AWS Access Key ID:     <your-key-id>
   AWS Secret Access Key: <your-secret-key>
   Default region name:   us-east-2        ← must match the bucket region
   Default output format: json
   ```

   This writes credentials to `~/.aws/credentials`, which DVC reads automatically.

   Verify access:
   ```bash
   aws s3 ls s3://trading-management-dvc/
   # Should list files without an error
   ```

   > **Alternative — environment variables** (CI / Docker):
   > ```bash
   > export AWS_ACCESS_KEY_ID=your-key-id
   > export AWS_SECRET_ACCESS_KEY=your-secret-key
   > export AWS_DEFAULT_REGION=us-east-2
   > ```

---

## Step 1 — Pull the Model from DVC

Run from the **repo root** (`trading_management/`):

```bash
# macOS / Linux
dvc pull langgraph/backtest/data/models/qlora_cloud.dvc

# Windows (PowerShell)
dvc pull langgraph/backtest/data/models/qlora_cloud.dvc
```

This downloads ~4.4 GB from S3 into:
```
langgraph/backtest/data/models/qlora_cloud/gguf_gguf/Qwen2.5-7B-Instruct.Q4_K_M.gguf
```

---

## Step 2 — Copy the Modelfile

The Modelfile is tracked in git at `optimization/qlora/Modelfile`.
Copy it next to the GGUF:

```bash
# macOS / Linux
cp langgraph/optimization/qlora/Modelfile \
   langgraph/backtest/data/models/qlora_cloud/gguf_gguf/Modelfile

# Windows (PowerShell)
Copy-Item langgraph\optimization\qlora\Modelfile `
          langgraph\backtest\data\models\qlora_cloud\gguf_gguf\Modelfile
```

---

## Step 3 — Register the Model in Ollama

Navigate to the GGUF directory and create the model:

```bash
# macOS / Linux
cd langgraph/backtest/data/models/qlora_cloud/gguf_gguf
ollama create trading-qwen-ft -f Modelfile

# Windows (PowerShell)
cd langgraph\backtest\data\models\qlora_cloud\gguf_gguf
ollama create trading-qwen-ft -f Modelfile
```

Ollama will load the GGUF and register it as `trading-qwen-ft`.
This takes ~30 seconds and only needs to be done once.

Verify it's registered:
```bash
ollama list
# Should show: trading-qwen-ft
```

---

## Step 4 — Test It

Quick smoke test — send it a raw prompt:

```bash
ollama run trading-qwen-ft "Reply with only the word: READY"
```

More realistic test — mimic the trading agent's prompt format:

```bash
ollama run trading-qwen-ft \
  "You are a Senior Technical Analyst. Given: Symbol: BTCUSDT, RSI: 42, ADX: 31, EMA aligned bullish. Output JSON with bias, entry, tp, sl, reasoning, confidence."
```

Expected: a valid JSON response with `"bias": "LONG"` or `"SHORT"`.

---

## Step 5 — Wire It Into the LangGraph Agent

In `langgraph/.env` (create it if it doesn't exist):

```env
LLM_PROVIDER=ollama
LLM_MODEL=trading-qwen-ft
OLLAMA_BASE_URL=http://localhost:11434
```

Then restart the agent:

```bash
cd langgraph
source .venv/bin/activate
uvicorn agent.main:app --port 2024 --reload
```

The agent will now use the fine-tuned model for all `generator_node()` calls instead of the base `qwen2.5:14b`.

---

## Switching Back to the Base Model

To revert to the original 14B model:

```env
LLM_MODEL=qwen2.5:14b
```

No changes to code needed — just the env var.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ollama: command not found` | Ollama not installed | Install from ollama.com |
| `error loading model` | GGUF file missing or corrupt | Re-run `dvc pull` |
| `out of memory` | Not enough RAM | Close other apps; or use CPU-only mode |
| Responses not JSON | Temperature too high | Modelfile already sets `temperature 0.1` — check it wasn't overwritten |
| Very slow (~1 tok/s) | Running on CPU with low RAM | Normal for CPU-only; add a GPU or use cloud |
| `trading-qwen-ft not found` | `ollama create` not run yet | Run Step 3 again |

---

## What This Model Knows

The fine-tuned model was trained on **39,312 labeled crypto trading samples** (18 months,
12 symbols, 3 timeframes) via QLoRA on top of Qwen 2.5 7B Instruct. It learned:

- Which market conditions signal LONG vs SHORT (88% direction accuracy on test set)
- How to set entry, TP, and SL levels relative to ATR (61.7% win rate)
- How to output structured JSON consistently in the trading agent format

It was evaluated against the same base model (zero-shot Qwen 2.5 7B) and outperformed
it by **+29.5 percentage points** in direction accuracy (McNemar p ≈ 0, chi² = 557).
