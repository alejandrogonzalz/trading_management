# QLoRA Fine-Tuning on Windows (RTX 5070 Ti 16GB)

Step-by-step guide to fine-tune Qwen 2.5 7B locally on an **RTX 5070 Ti** (16GB
VRAM) running Windows with PowerShell. This is a cheaper second data point to the
cloud model: same fixed strict-temporal split, 1 epoch capped at 1500 steps (~14h
overnight, $0 compute).

> The cloud guide is `RUNPOD_GUIDE.md`. This is the local counterpart. Same dataset,
> same split, different hardware constraints: batch=1, no FlashAttention-2, Triton
> kernels only.

---

## Cost analysis (local vs cloud)

| | Local (RTX 5070 Ti) | RunPod A100 80GB |
|---|---|---|
| GPU | RTX 5070 Ti 16GB | NVIDIA A100 80GB |
| Compute cost | **$0** | ~$5–6 |
| FlashAttention-2 | No (Triton) | Yes (pre-installed) |
| Throughput | ~28 s/step | ~1.5–2 s/step (FA2, batch=8) |
| Batch size | 1 | 8 |
| Training (2 epochs) | N/A (1 epoch, 1500 steps) | ~3–3.5 h |
| Eval (test set strided) | ~2 h (200 samples) | ~10–15 min (3000 batched) |
| Total | **~14 h** | ~4 h |
| Tradeoff | Ties up the workstation overnight | Costs ~$5, stop the pod when done |

Local is free but slow. Run overnight (disable sleep) and use the machine normally
the next morning. The cloud model is the primary result; local is optional validation.

---

## Prerequisites

Before starting, verify all of the following on your Windows machine.

### 1. NVIDIA driver (R570+ for Blackwell sm_120)

```powershell
nvidia-smi
```

Must show driver version >= 572.xx. If not, update from
https://www.nvidia.com/download/index.aspx (Game Ready or Studio, either works).

### 2. Python 3.10+ installed

```powershell
python --version
```

Needs 3.10, 3.11, or 3.12. Install from https://www.python.org/downloads/ if missing.
Ensure "Add to PATH" is checked during install.

### 3. Git (for cloning Unsloth)

```powershell
git --version
```

Install from https://git-scm.com/download/win if missing.

### 4. VRAM free (nothing else using the GPU)

```powershell
nvidia-smi
```

Must show 0 MiB used (or close). Close Ollama, Chrome (GPU acceleration), games,
any ML processes. 16GB is tight -- you need every MB.

### 5. Disk space (~30 GB free)

| What | Size |
|------|------|
| HuggingFace model cache (first download) | ~5 GB |
| Checkpoints (3 rotating) | ~15 GB |
| GGUF export | ~4 GB |
| Dataset + candles | ~2 GB |
| Headroom | ~4 GB |

### 6. AWS CLI (for DVC pull)

```powershell
aws --version
aws sts get-caller-identity
```

Needed to pull dataset + candles from S3. Install from
https://aws.amazon.com/cli/ if missing. Run `aws configure` with credentials that
have `s3:GetObject` on `s3://trading-management-dvc/`.

---

## Step 1: Setup (one-time, ~15-20 min)

Run the setup script from the `langgraph/` directory:

```powershell
cd C:\Users\alex\projects\trading_management\langgraph
powershell -ExecutionPolicy Bypass -File optimization\qlora\setup_local.ps1
```

Or if execution policy allows:

```powershell
.\optimization\qlora\setup_local.ps1
```

This creates `.venv-finetuning/`, installs Unsloth + dependencies, and verifies
CUDA + GPU. See the script for details.

**If setup_local.ps1 fails**, do it manually:

```powershell
cd C:\Users\alex\projects\trading_management\langgraph

# Create venv
python -m venv .venv-finetuning
.\.venv-finetuning\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip

# Install Unsloth (resolves its own torch + TRL + transformers + peft)
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Install remaining deps
pip install "dvc[s3]" scikit-learn pyyaml tqdm matplotlib httpx

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "from unsloth import FastLanguageModel; print('Unsloth OK')"
```

---

## Step 2: Pull data (REQUIRED)

The dataset and candles are tracked by DVC in S3. Without candles, trade simulation
metrics (win_rate, profit_factor, Sharpe) come out as 0.

```powershell
cd C:\Users\alex\projects\trading_management\langgraph
.\.venv-finetuning\Scripts\Activate.ps1
dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles
```

Verify:

```powershell
python -c "print(sum(1 for _ in open('backtest/data/labeled/dataset.jsonl')))"
# Must print: 56161
```

**If DVC is not configured**, pull directly with the AWS CLI:

```powershell
aws s3 cp s3://trading-management-dvc/files/md5/ backtest\data\ --recursive
# Or use the exact DVC cache paths (check .dvc files for the MD5 hashes)
```

---

## Step 3: Smoke test (~5-10 min)

Before committing to a 14h run, verify everything works end-to-end:

```powershell
.\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py `
  --max-steps 3 --max-eval 10 --batch-size 1 --grad-accum 16 `
  --rank 8 --alpha 16 --diagnostic-samples 0 `
  2>&1 | Tee-Object -FilePath optimization\qlora\logs\smoke_test.log
```

What to check:
- Model loads without error (~2 min first time, downloads ~5GB)
- 3 training steps complete without OOM
- `RESULT:` line appears at the end
- `nvidia-smi` during training shows ~10-13 GB of 16 GB used (not 16/16)

If step 0 passes, the full run will work.

---

## Step 4: Prevent sleep (IMPORTANT for overnight runs)

Windows will sleep mid-training if power settings allow it. Fix before launching:

### Option A: Windows Settings (recommended)

Settings > System > Power & battery > Screen and sleep:
- Set "When plugged in, put my device to sleep after" to **Never**
- Restore after training completes.

### Option B: powercfg (command line)

```powershell
# Disable sleep (run as admin)
powercfg /change standby-timeout-ac 0
powercfg /change standby-timeout-dc 0

# Re-enable after (e.g., 30 minutes)
powercfg /change standby-timeout-ac 30
```

### Option C: caffeinate equivalent

```powershell
# In a separate terminal — keeps the system awake by simulating input
while ($true) { [System.Threading.Thread]::Sleep(60000); [System.Windows.Forms.SendKeys]::SendWait('{SCROLLLOCK}') }
```

(Requires `Add-Type -AssemblyName System.Windows.Forms` first.)

---

## Step 5: Run the real training

Use the `run_local.ps1` wrapper — it cd's to the langgraph root, runs `train_qlora.py`
with the documented local config, creates the log dir, and tees all output to
`optimization\qlora\logs\qlora_local.log`:

```powershell
cd C:\Users\alex\projects\trading_management\langgraph
.\optimization\qlora\run_local.ps1
```

This is the PowerShell equivalent of `run_local.sh`. The raw command it runs is:

```powershell
.\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py `
  --lr 0.00005 --rank 8 --alpha 16 --epochs 1 --max-steps 1500 `
  --batch-size 1 --grad-accum 16 --max-eval 200 --diagnostic-samples 0 `
  --tag qlora_local --output-dir backtest\data\models\qlora_local `
  2>&1 | Tee-Object -FilePath optimization\qlora\logs\qlora_local.log
```

### Keep it alive after closing the terminal (the "tmux" equivalent)

Windows has no tmux, but `Start-Process` launches the run **detached** — it keeps
going after you close the launching terminal. Because `run_local.ps1` always tees to
`qlora_local.log`, the log is written whether or not a window is visible:

```powershell
cd C:\Users\alex\projects\trading_management\langgraph
Start-Process powershell -WindowStyle Hidden -ArgumentList `
  '-NoProfile','-ExecutionPolicy','Bypass','-File', `
  'C:\Users\alex\projects\trading_management\langgraph\optimization\qlora\run_local.ps1'
```

Then follow the log from any terminal (this is your `tmux attach`; `Ctrl+C` only stops
the tail, not the training):

```powershell
Get-Content C:\Users\alex\projects\trading_management\langgraph\optimization\qlora\logs\qlora_local.log -Wait -Tail 40
```

**Caveat vs tmux:** a detached process survives closing the terminal but **not** a
full Windows log-off, restart, or sleep. Stay logged in and keep sleep disabled
(Step 4) for the ~14h run. If it does die, resume from the last checkpoint (Step 7).

If you prefer to watch it live instead, just run `.\optimization\qlora\run_local.ps1`
in a terminal and leave it open overnight — same log file either way.

### What happens during training

```
Phase 1 — Load model:         ~2 min (from HF cache after first download)
Phase 2 — Prepare data:       ~1 min (temporal split + chat-format tokenization)
Phase 3 — Train (1500 steps): ~11.7 h (28s/step average, variable 9-65s)
Phase 4 — Save model + GGUF:  ~15 min
Phase 5 — Evaluate:           ~2 h (200 test samples x 35s each)
```

### Healthy signals during training

| Metric | Expected | Problem if |
|--------|----------|-----------|
| s/step | ~25-35s | >60s (something competing for GPU) |
| GPU memory | ~10-13 GB / 16 GB | 16/16 (OOM imminent) |
| GPU utilization | 90-100% | <50% (CPU/disk bottleneck) |
| Loss (first 100 steps) | ~1.5-2.5, decreasing | >5 or NaN (bad LR or data) |
| Loss (step 500+) | ~0.7-1.2, stable | Spiking (instability) |

---

## Step 6: Monitor (in a separate PowerShell window)

```powershell
# GPU utilization every 10 seconds
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total --format=csv -l 10

# Follow the log file (like tail -f)
Get-Content optimization\qlora\logs\qlora_local.log -Wait -Tail 20

# Check if the process is still running
Get-Process python* | Format-Table Id, CPU, WorkingSet64
```

---

## Step 7: If interrupted (crash, reboot, power loss)

Checkpoints save every 250 steps. Resume from the last one — `run_local.ps1`
forwards any extra flags to `train_qlora.py`, so just append `--resume`:

```powershell
.\optimization\qlora\run_local.ps1 --resume
```

(Detached: add `--resume` as a final element in the `Start-Process` `-ArgumentList`.)

`--resume` finds the last `checkpoint-N` in `--output-dir` and continues. No work
is lost -- training picks up at the exact step where it stopped.

**Warning:** if you previously ran a smoke test, its `checkpoint-3` may still be in
the output dir. Delete it before `--resume` to avoid resuming from it:

```powershell
Remove-Item -Recurse backtest\data\models\qlora_local\checkpoint-3
```

---

## Step 8: When done

The script prints a summary:

```
RESULT: Test Accuracy = X.XXXX
Win Rate = X.XXXX  Profit Factor = X.XXX
Sharpe = X.XXX  Max Drawdown = X.XX%
Time = XXXXs
```

Output files:

| File | Content |
|------|---------|
| `optimization/qlora/results/qlora_local.json` | Full result with metrics + predictions |
| `optimization/qlora/results/qlora_optimization.json` | Canonical copy (latest model) |
| `optimization/qlora/results/qlora_local_loss_curve.json` | Loss data (for plotting) |
| `optimization/qlora/results/qlora_local_loss_curve.png` | Loss plot |
| `backtest/data/models/qlora_local/` | LoRA adapters + merged GGUF |

### Backup with DVC + git

```powershell
# Track model weights (too large for git)
dvc add backtest\data\models\qlora_local
dvc push

# Commit pointer + results
git add backtest\data\models\qlora_local.dvc backtest\data\models\.gitignore
git add optimization\qlora\results\qlora_local.json
git add optimization\qlora\results\qlora_local_loss_curve.json
git add optimization\qlora\results\qlora_local_loss_curve.png
git commit -m "feat(qlora): local fine-tuning results on fixed split (RTX 5070 Ti)"
git push origin fix/qlora-data-leak
```

### Re-enable sleep

```powershell
# If you disabled via Settings: revert to your preferred timeout
# If you used powercfg:
powercfg /change standby-timeout-ac 30
```

---

## Step 9: Compare results

```powershell
cd C:\Users\alex\projects\trading_management\langgraph
.\.venv-finetuning\Scripts\python.exe -m cli compare-stats `
  --a optimization/qlora/results/qlora_local.json `
  --b backtest/data/results/ml-lstm.json
```

Or compare cloud vs local:

```powershell
.\.venv-finetuning\Scripts\python.exe -m cli compare-stats `
  --a optimization/qlora/results/qlora_cloud.json `
  --b optimization/qlora/results/qlora_local.json
```

---

## Optional: Deploy to Ollama

After training, the GGUF model can be loaded into Ollama for LangGraph integration:

```powershell
# Create a Modelfile
@"
FROM ./backtest/data/models/qlora_local/gguf/unsloth.Q8_0.gguf
PARAMETER temperature 0.1
SYSTEM "You are a Senior Technical Analyst..."
"@ | Set-Content -Path backtest\data\models\qlora_local\Modelfile

# Import into Ollama
ollama create trading-qwen-ft-local -f backtest\data\models\qlora_local\Modelfile
```

Then set `LLM_MODEL=trading-qwen-ft-local` in `.env` -- the LangGraph agent picks
it up via `llm_factory.py`.

---

## Troubleshooting

| Issue | Likely Cause | Fix |
|-------|-------------|-----|
| OOM on step 0 | Other process using GPU | Close Ollama, Chrome, games. Check `nvidia-smi` |
| OOM on step N (not 0) | VRAM fragmentation | Kill process, verify GPU clear, re-run with `--resume` |
| Very slow (>60s/step) | GPU contention | Close competing processes. Check Task Manager GPU tab |
| PC slept mid-training | Sleep not disabled | See Step 4. Then `--resume` |
| `CUDA: False` | Wrong PyTorch build | Reinstall: `pip install torch --index-url https://download.pytorch.org/whl/cu128 --upgrade` |
| `sm_120` not supported | Old bitsandbytes | Upgrade: `pip install bitsandbytes>=0.45` |
| Unsloth import fails | Dependency conflict | Recreate venv from scratch (run `setup_local.ps1`) |
| Loss = NaN | LR too high | Reduce to `--lr 0.00002` |
| Loss not decreasing after 500 steps | Dataset issue | Verify dataset has 56161 lines; check DVC pull succeeded |
| `checkpoint-3` confuses resume | Leftover smoke test | Delete it: `Remove-Item -Recurse ...\checkpoint-3` |
| GGUF stage hangs (~15 min) | Normal merge + quantization | Not a crash. Wait for it |
| "Python.h not found" | Missing dev headers | Not applicable on Windows (Triton handles compilation differently) |
| Triton compilation errors | Stale cache | Delete `unsloth_compiled_cache/` and retry |

---

## Version requirements (RTX 5070 Ti / Blackwell sm_120)

The RTX 5070 Ti uses the **Blackwell** architecture (`sm_120`). This is new hardware
and requires specific minimum versions. If any are too old, training will crash with
CUDA errors or unrecognized GPU messages.

| Component | Minimum Version | Why | How to Check |
|-----------|----------------|-----|-------------|
| NVIDIA driver | R570+ (Windows: >= 572.xx) | `sm_120` support added in R570 | `nvidia-smi` (top row) |
| PyTorch | >= 2.7, **cu128** build | First stable release with `sm_120` kernels | `python -c "import torch; print(torch.__version__, torch.version.cuda)"` |
| CUDA (bundled in PyTorch) | 12.8 | Blackwell supported from CUDA 12.8 | Same as above (torch.version.cuda) |
| bitsandbytes | >= 0.45 | 4-bit quantization on `sm_120` added there | `pip show bitsandbytes` |
| Triton | Recent `triton-windows` or `triton-nightly` | Unsloth uses Triton (no FA2 on Windows) | Installed by Unsloth automatically |

Verify all at once:

```powershell
.\.venv-finetuning\Scripts\python.exe -c @"
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'Compute capability: {torch.cuda.get_device_capability(0)}')
import bitsandbytes; print(f'bitsandbytes: {bitsandbytes.__version__}')
from unsloth import FastLanguageModel; print('Unsloth: OK')
"@
```

Expected output:
```
PyTorch: 2.7.x+cu128
CUDA: 12.8
GPU: NVIDIA GeForce RTX 5070 Ti
Compute capability: (12, 0)
bitsandbytes: 0.45.x
Unsloth: OK
```

---

## Training config explained

| Param | Value | Rationale |
|-------|-------|-----------|
| `--lr 0.00005` | 5e-5 | Higher LR compensates for only 1 epoch (fewer total updates) |
| `--rank 8` | 8 | Smaller adapters (~20M params). Fits 16GB comfortably |
| `--alpha 16` | 16 | Standard 2x rank scaling |
| `--epochs 1` | 1 | One pass, capped at 1500 steps |
| `--max-steps 1500` | 1500 | ~60% of a full epoch. Learns main patterns without full 19h |
| `--batch-size 1` | 1 | Only value that fits 1024-token samples in 16GB |
| `--grad-accum 16` | 16 | Effective batch = 1 x 16 = 16 (matches cloud) |
| `--max-eval 200` | 200 | ~18 samples/symbol. Light but representative |
| `--diagnostic-samples 0` | 0 | Skip train/val gap probes (saves ~2h). Cloud handles full diag |
| `--tag qlora_local` | - | Names all output files |

---

## Single model, not a sweep

The thesis trains **one** defensible model per infrastructure approach. The old
multi-config sweep was invalidated by the leakage fix and archived under
`results/archive/`. This local run produces a single result that pairs with the
cloud model for comparison.
