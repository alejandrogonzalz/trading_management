# Local Fine-Tuning via Docker — RTX 5070 Ti

Plan for running `unsloth/unsloth:latest` locally on Windows using Docker Desktop + WSL2
to get FlashAttention-2 on the RTX 5070 Ti without a full Linux install.

---

## Why Docker Instead of the Native Windows Path

The current local setup (`run_local.sh` in `.venv-finetuning/`) runs ~28s/step because
FlashAttention-2 is not available on Windows. Unsloth falls back to Triton kernels, which
are slower and have higher peak VRAM. The log banner shows:

```
FA [Xformers = None. FA2 = False]   ← current Windows native result
```

`unsloth/unsloth:latest` is the exact same image used on RunPod — it ships with
FA2 2.8.3 compiled for CUDA 12.8. Running it locally via Docker Desktop + GPU passthrough
gives the same Linux environment on Windows hardware with zero additional installation.

Expected result with FA2 active:
```
FA [Xformers = None. FA2 = True]    ← target Docker result
```

### Performance comparison

| Setup | FA2 | Batch | Step time | Epoch time (2452 steps) | Cost |
|-------|-----|-------|-----------|------------------------|------|
| Windows native (`.venv-finetuning`) | No | 1 | ~28s | ~19h | $0 |
| **Docker local (RTX 5070 Ti)** | **Yes** | **1–2** | **~5–12s** | **~3–8h** | **$0** |
| RunPod A100 80GB | Yes | 2 | ~3.6s | ~2.5h | ~$3.50/epoch |

FA2 typically gives 2–4× speedup on 1024-token sequences — the dominant bottleneck
at step time. The RTX 5070 Ti has ~2–3× less raw compute than an A100, so expect
~5–12s/step rather than the A100's 3.6s/step.

---

## Prerequisites

### 1. Docker Desktop (Windows, WSL2 backend)

Docker Desktop must use the WSL2 backend (not Hyper-V) to enable GPU passthrough.

```
Docker Desktop → Settings → General
  ✓ Use WSL 2 based engine
```

Check the installed version: `docker --version` — any version ≥ 24 is fine.

### 2. NVIDIA GPU driver on the Windows host

The container uses the Windows driver via the NVIDIA Container Toolkit bridge —
you do **not** install CUDA inside WSL2. Minimum driver version: **≥ 530** for CUDA 12.x.

```powershell
nvidia-smi   # must show the GPU and driver version
```

### 3. NVIDIA Container Toolkit enabled in Docker Desktop

```
Docker Desktop → Settings → Docker Engine
```

Verify the `nvidia` runtime is available:
```powershell
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

This should print the RTX 5070 Ti with driver info. If it fails, check:
- Docker Desktop is on the latest version
- The NVIDIA driver is current
- `--gpus all` support is enabled in Docker Desktop settings

> **RTX 5070 Ti architecture note:** If the GPU is Blackwell (`sm_100`), verify that
> FA2 2.8.3 in the image supports it. Check the startup banner when you first run the
> container:
> ```
> unsloth INFO: FA2 = True   ← good
> unsloth INFO: FA2 = False  ← FA2 compiled for an older arch; Triton fallback still faster than Windows
> ```
> Even without FA2, the Linux environment gives ~15–18s/step (better than Windows 28s).

---

## The docker run command

Run from the project root on Windows (PowerShell):

```powershell
# From: C:\Users\alex\projects\trading_management\langgraph

docker run --rm `
  --gpus all `
  --ipc=host `
  --ulimit memlock=-1 `
  -v "${PWD}:/workspace" `
  -w /workspace `
  docker.io/unsloth/unsloth:latest `
  bash optimization/qlora/run_docker_local.sh
```

Or from WSL2 terminal (better file I/O if project is in WSL2 filesystem):
```bash
# From: ~/projects/trading_management/langgraph  (WSL2 path, not /mnt/c/...)
docker run --rm \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  -v "$(pwd):/workspace" \
  -w /workspace \
  docker.io/unsloth/unsloth:latest \
  bash optimization/qlora/run_docker_local.sh
```

> **File I/O performance tip:** Docker on Windows is fastest when the project lives in
> the WSL2 filesystem (`~/projects/...`) rather than the Windows filesystem
> (`/mnt/c/Users/...`). For training, only the initial data load matters — subsequent
> GPU compute is unaffected by filesystem location. Either path works.

### Flag meanings

| Flag | Why |
|------|-----|
| `--gpus all` | Passes the RTX 5070 Ti through to the container |
| `--ipc=host` | Shared memory for PyTorch DataLoader workers — prevents `Bus error` |
| `--ulimit memlock=-1` | Allows the GPU to lock memory; required by CUDA for large allocations |
| `-v "$(pwd):/workspace"` | Mounts the `langgraph/` directory into the container |
| `-w /workspace` | Sets the working directory inside the container to the mounted path |
| `--rm` | Cleans up the container after training completes |

---

## run_docker_local.sh

Script: `optimization/qlora/run_docker_local.sh`

This runs inside the container. The script:
1. Runs a smoke test (3 steps) to confirm FA2 and VRAM fit, then exits early if the
   smoke test fails — so you don't waste hours on a broken setup
2. If the smoke test passes, launches the full single-epoch training run

```bash
#!/usr/bin/env bash
# Run INSIDE unsloth/unsloth:latest container
# Mount: -v "$(pwd):/workspace" -w /workspace
# GPU:   --gpus all --ipc=host --ulimit memlock=-1

set -e

LOG_DIR="optimization/qlora/logs"
mkdir -p "$LOG_DIR"

echo "=== Docker local training on RTX 5070 Ti ==="
echo "=== FA2 check — look for 'FA2 = True' in the banner ==="

# ── smoke test (3 steps, tiny eval) ──────────────────────────────────────────
echo ""
echo "--- Smoke test (3 steps) ---"
python3 optimization/qlora/train_qlora.py \
  --lr 5e-5 \
  --rank 8 \
  --alpha 16 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 16 \
  --tag qlora_docker_local \
  --max-steps 3 \
  --max-eval 20 \
  2>&1 | tee "$LOG_DIR/smoke_docker_local.log"

echo ""
echo "--- Smoke test passed. Launching full 1-epoch run ---"
echo "--- Monitor with: python3 optimization/qlora/monitor_training.py --log $LOG_DIR/run_docker_local.log --tz -6 --epochs 1 ---"
echo ""

# ── full single-epoch run ─────────────────────────────────────────────────────
python3 optimization/qlora/train_qlora.py \
  --lr 5e-5 \
  --rank 8 \
  --alpha 16 \
  --epochs 1 \
  --batch-size 1 \
  --grad-accum 16 \
  --tag qlora_docker_local \
  2>&1 | tee "$LOG_DIR/run_docker_local.log"

echo ""
echo "=== Done. Result: optimization/qlora/results/qlora_docker_local.json ==="
```

### Why these hyperparameters

| Param | Value | Reason |
|-------|-------|--------|
| `lr` | 5e-5 | Slightly higher than cloud (2e-5) to learn faster in 1 epoch |
| `rank` | 8 | Half the cloud rank; reduces VRAM pressure on 16GB |
| `alpha` | 16 | 2× rank (standard scaling) |
| `epochs` | 1 | 1 epoch ≈ 3–8h; enough to verify the model learns before committing overnight |
| `batch-size` | 1 | Conservative baseline for 16GB; bump to 2 only after smoke test confirms fit |
| `grad-accum` | 16 | Effective batch = 16; matches cloud effective batch |

**Trying batch-size=2:** if the smoke test passes cleanly, edit the script and re-run
`--max-steps 3` with `--batch-size 2`. If it completes without CUDA out-of-memory,
batch=2 is safe and cuts epoch time roughly in half (~1.5–4h for 1 epoch).

---

## Monitoring the run

From a second PowerShell or WSL2 terminal:

```powershell
# Stream the log live
Get-Content -Wait "optimization\qlora\logs\run_docker_local.log"

# Or use the monitor script (once at least 1 checkpoint is written)
.\.venv\Scripts\python.exe optimization\qlora\monitor_training.py `
  --log optimization\qlora\logs\run_docker_local.log `
  --epochs 1 `
  --tz -6 `
  --watch
```

Check GPU utilization in another terminal:
```powershell
nvidia-smi -l 5   # refresh every 5 seconds
```

---

## VRAM budget

With FA2 at 1024 tokens, batch=1, rank=8:

| Component | Estimated VRAM |
|-----------|---------------|
| Model weights (4-bit base + bf16 adapters) | ~5.5 GB |
| Activations at batch=1, 1024 tokens, 28 layers | ~5–6 GB |
| Optimizer state (Adam moments × adapter params) | ~0.5 GB |
| FA2 attention buffer (1024 tokens × 28 heads) | ~0.5 GB |
| **Total** | **~11.5–12.5 GB** |
| Headroom (of 16 GB) | ~3.5–4.5 GB |

Attempting batch=2 adds ~5–6 GB of activations → total ~16.5–18 GB → likely OOM.
FA2 reduces peak activations by ~30–40%, so batch=2 might land at ~14–15 GB and fit.
Test with `--max-steps 3 --batch-size 2` before committing.

---

## How the result compares to the cloud run

The Docker local run uses a different tag (`qlora_docker_local`) and different
hyperparameters (rank=8, lr=5e-5, 1 epoch) than the cloud run (rank=16, lr=2e-5, 2
epochs). They are not directly comparable configurations, but the local result is
useful as:

1. **A sanity check** that the training loop works with the fixed split before waiting
   for the cloud result
2. **A second data point** for the thesis: local GPU vs cloud GPU cost-accuracy trade-off
3. **A recovery run** if the RunPod result is lost or corrupted

For the primary comparison table in `Avance6.ipynb`, use the cloud result
(`qlora_optimization.json`) — it has more epochs and higher rank. The Docker local
result can appear in a supplementary table or in the cloud infrastructure analysis section.

---

## End-to-end checklist

```
[ ] docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi  → GPU visible
[ ] docker pull docker.io/unsloth/unsloth:latest                                → image pulled
[ ] Smoke test passes (3 steps, no OOM, FA2 banner shows True)
[ ] Optional: retry smoke test with --batch-size 2 (check OOM)
[ ] Full run launched (nohup or terminal you can leave open)
[ ] monitor_training.py shows loss decreasing after first checkpoint
[ ] Run completes → optimization/qlora/results/qlora_docker_local.json exists
[ ] win_rate > 0 in result JSON (candles were mounted via the volume)
[ ] dvc add backtest/data/models/qlora_docker_local && dvc push
```

---

## Files referenced

| File | Role |
|------|------|
| `optimization/qlora/run_docker_local.sh` | Script that runs inside the container |
| `optimization/qlora/run_cloud.sh` | Equivalent cloud script (for comparison) |
| `optimization/qlora/train_qlora.py` | Training script called by both wrappers |
| `optimization/qlora/monitor_training.py` | Live log parser with UTC-6, warnings |
| `optimization/qlora/logs/run_docker_local.log` | Output log (created by the script) |
| `optimization/qlora/results/qlora_docker_local.json` | Result JSON from this run |
| `docs/QLORA_FINETUNING.md` | Full QLoRA status, constraints, runbook |
| `docs/HYPERPARAMETERS.md` | Explanation of every hyperparameter chosen above |
| `docs/NEXT_STEPS.md` | Where this run fits in the thesis pipeline |
