# QLoRA Local Training — Docker on Windows GPU

Runs the same `docker.io/unsloth/unsloth:latest` image used on RunPod, locally on a
Windows machine with NVIDIA GPU. FlashAttention-2 is pre-installed in the image — no
venv, no manual dependency setup, no platform compatibility issues.

**When to use**: free second data point alongside the cloud model, or to run additional
configs (aggressive/conservative) without paying for RunPod hours. Same fixed temporal
split, same training script, same evaluation — just slower (~14h per epoch on 16GB VRAM
vs ~2h on A100 80GB).

---

## Prerequisites

### 1. Windows + WSL2

WSL2 must be enabled. Docker Desktop uses it as the backend for GPU passthrough.

```powershell
# Check WSL2 is installed (PowerShell, elevated):
wsl --list --verbose
# Should show a distro (Ubuntu recommended) with VERSION 2
```

If not installed: `wsl --install -d Ubuntu` and restart.

### 2. Docker Desktop

Install Docker Desktop for Windows. In Settings:
- General → "Use the WSL 2 based engine" ✓
- Resources → WSL Integration → Enable for your distro (Ubuntu) ✓

### 3. NVIDIA GPU Driver (Windows side)

Install the latest Game Ready or Studio driver from [nvidia.com/drivers](https://www.nvidia.com/drivers). Must be **R570+** (version ≥ 572.xx on Windows).

```powershell
# Verify (PowerShell):
nvidia-smi
# Should show your GPU + driver version
```

### 4. NVIDIA Container Toolkit (WSL2 side)

This enables Docker containers to access the GPU. Run inside WSL2 (Ubuntu):

```bash
wsl -d Ubuntu

# Add NVIDIA package repo:
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 5. Verify GPU access from Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi
```

Must show your GPU. If it errors with "could not select device driver", the container
toolkit is not installed correctly — repeat step 4.

### 6. Disk space and GPU availability

| Requirement | Minimum |
|-------------|---------|
| Free disk | ~30 GB (image 25GB + checkpoints 15GB + GGUF 4GB) |
| GPU VRAM free | All of it — close Ollama, games, Chrome GPU accel |
| AWS CLI (for DVC) | `aws sts get-caller-identity` must work |

---

## Step 1 — Pull the image (one-time, ~25 GB)

```bash
docker pull docker.io/unsloth/unsloth:latest
```

This contains: Python 3.12, PyTorch 2.10+cu128, flash-attn 2.8.3, unsloth,
bitsandbytes, TRL, transformers — everything pre-compiled.

---

## Step 2 — Pull data on the HOST (REQUIRED, before Docker)

DVC runs on the **host** (Windows), not inside the container. The container just reads
the files via the bind mount. Without candles, `win_rate` / `profit_factor` / `Sharpe`
come out 0.

```powershell
# PowerShell (Windows) — from the repo root (trading_management/), NOT langgraph/
cd C:\Users\alex\projects\trading_management
pip install dvc[s3]  # one-time, if not already installed
aws configure        # one-time, if not already configured

dvc pull langgraph/backtest/data/labeled/dataset.jsonl langgraph/backtest/data/candles
```

Or from WSL2:
```bash
cd ~/projects/trading_management
dvc pull langgraph/backtest/data/labeled/dataset.jsonl langgraph/backtest/data/candles
```

Verify:
```bash
wc -l langgraph/backtest/data/labeled/dataset.jsonl
# Must print: 56161

ls langgraph/backtest/data/candles/*.json | wc -l
# Must print: 36 (12 symbols × 3 timeframes)
```

> **Important**: DVC is configured at the repo root (`trading_management/.dvc/config`),
> not at `langgraph/`. Always run `dvc pull` from the repo root.

---

## Step 3 — Run training

Three options, pick whichever fits your workflow:

### Option A: Docker Compose (simplest)

```bash
cd langgraph
docker compose -f optimization/qlora/docker-compose.local.yml up
```

With custom config:
```bash
QLORA_FLAGS="--tag config2 --lr 0.00005 --rank 8 --epochs 3" \
  docker compose -f optimization/qlora/docker-compose.local.yml up
```

### Option B: PowerShell script

```powershell
cd C:\Users\alex\projects\trading_management\langgraph

# Default config (lr=5e-5, rank=8, 1 epoch):
.\optimization\qlora\run_docker_windows.ps1

# Custom config:
.\optimization\qlora\run_docker_windows.ps1 --tag config3 --lr 0.00001 --rank 32 --alpha 64 --epochs 2

# Interactive shell (debugging):
.\optimization\qlora\run_docker_windows.ps1 -Shell
```

### Option C: Manual docker run (WSL2)

```bash
cd ~/projects/trading_management/langgraph

# Foreground (see output directly):
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "$(pwd):/workspace" -w /workspace \
  docker.io/unsloth/unsloth:latest \
  bash optimization/qlora/run_docker_local.sh

# Custom config:
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "$(pwd):/workspace" -w /workspace \
  docker.io/unsloth/unsloth:latest \
  bash optimization/qlora/run_docker_local.sh --tag config2 --lr 0.00005 --rank 8 --epochs 3

# Detached (survives closing terminal):
nohup docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "$(pwd):/workspace" -w /workspace \
  docker.io/unsloth/unsloth:latest \
  bash optimization/qlora/run_docker_local.sh \
  > optimization/qlora/logs/run_local.log 2>&1 &
echo "PID: $!"
```

### What happens when you launch

1. Container starts, runs `nvidia-smi` (confirm GPU visible)
2. Prints PyTorch + FA2 versions (confirm `FlashAttention-2: 2.8.x`)
3. Installs missing project deps (dvc, sklearn, xgboost — ~30s)
4. **Smoke test**: 3 training steps + 20-sample eval (catches OOM immediately)
5. **Full run**: trains for the configured epochs, evaluates, exports GGUF

**Check the startup banner** — must show `FA2 = True`. If it shows `FA2: not available`,
the image may be stale; run `docker pull docker.io/unsloth/unsloth:latest`.

---

## Available configs

| Config | Tag | LR | Rank | Alpha | Epochs | Use case |
|--------|-----|-----|------|-------|--------|----------|
| Default (aggressive) | `qlora_local` | 5e-5 | 8 | 16 | 1 | Quick overnight run, tests if task is easy |
| Conservative | `config3` | 1e-5 | 32 | 64 | 2 | More adapter capacity, slower learning |
| Match cloud | `config1_local` | 2e-5 | 16 | 32 | 2 | Same as RunPod config for reproducibility check |

Auto-detected batch sizes by VRAM:

| VRAM | batch_size | grad_accum | effective batch |
|------|-----------|------------|-----------------|
| ≥70 GB (A100) | 8 | 2 | 16 |
| ≥40 GB (L40S) | 2 | 8 | 16 |
| ≥20 GB (RTX 3090/4090) | 2 | 8 | 16 |
| <20 GB (RTX 5070 Ti 16GB) | 1 | 16 | 16 |

---

## Step 4 — Monitor (new terminal)

```bash
# Training progress (refreshes every 30s)
python3 optimization/qlora/monitor_training.py \
  --log optimization/qlora/logs/run_qlora_local.log --epochs 1 --watch

# Raw log tail
tail -f optimization/qlora/logs/run_qlora_local.log

# GPU stats (Windows PowerShell or WSL2)
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total \
  --format=csv -l 10
```

### Healthy signals (RTX 5070 Ti 16GB)

| Metric | Expected |
|--------|----------|
| s/step | ~25–35s (with FA2), ~60s (without) |
| GPU memory | ~10–13 GB / 16 GB |
| Loss (first 100 steps) | 1.5–2.5, decreasing |
| Loss (step 500+) | 0.7–1.2, stable |
| Total time (1 epoch) | ~14–19h |

---

## Step 5 — If interrupted

Checkpoints save every 250 steps. Resume by adding `--resume`:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "$(pwd):/workspace" -w /workspace \
  docker.io/unsloth/unsloth:latest \
  bash optimization/qlora/run_docker_local.sh --resume
```

Or with docker-compose:
```bash
QLORA_FLAGS="--resume" docker compose -f optimization/qlora/docker-compose.local.yml up
```

> If a smoke-test `checkpoint-3` is left over in the output dir, delete it first:
> `rm -rf backtest/data/models/qlora_local/checkpoint-3`

---

## Step 6 — Results

```
optimization/qlora/results/
├── qlora_qlora_local.json     ← metrics + predictions (paired via sample_keys)
├── qlora_local_loss_curve.json
└── qlora_local_loss_curve.png

backtest/data/models/qlora_local/
├── adapter_config.json
├── adapter_model.safetensors
└── gguf/unsloth.Q4_K_M.gguf   ← for Ollama deployment
```

Compare cloud vs local:
```bash
python3 -m cli compare-stats \
  --a optimization/qlora/results/qlora_cloud/result.json \
  --b optimization/qlora/results/qlora_qlora_local.json
```

---

## Step 7 — Deploy to Ollama (optional)

```bash
# Create Modelfile
cat > Modelfile << 'EOF'
FROM backtest/data/models/qlora_local/gguf/unsloth.Q4_K_M.gguf
PARAMETER temperature 0.1
PARAMETER num_ctx 1024
EOF

ollama create trading-qwen-ft-local -f Modelfile
ollama run trading-qwen-ft-local "Test prompt"

# Point the system to it:
# LLM_MODEL=trading-qwen-ft-local in .env
```

---

## Step 8 — Save to DVC + git

```bash
cd trading_management  # DVC root
dvc add langgraph/backtest/data/models/qlora_local/
dvc push
git add langgraph/backtest/data/models/qlora_local.dvc \
        langgraph/backtest/data/models/.gitignore \
        langgraph/optimization/qlora/results/
git commit -m "feat(qlora): local Docker training results (RTX 5070 Ti)"
git push
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `could not select device driver` | NVIDIA Container Toolkit not installed — redo prerequisite step 4 |
| OOM on step 0 | Close Ollama, Chrome, games — need ~3 GB free headroom above model |
| OOM on step N | VRAM fragmentation — stop container, `docker run` again with `--resume` |
| `FA2 = False` or `not available` | `docker pull docker.io/unsloth/unsloth:latest` (image may be stale) |
| Very slow (>60 s/step) | Something competing for GPU — check Task Manager → GPU tab |
| Loss = NaN | LR too high — use `--lr 0.00002` |
| Container exits immediately | Check output for CUDA / driver version mismatch. Driver must be R570+ |
| Slow I/O / pip installs hang | Project is on Windows FS (`/mnt/c/`). Move to WSL2 FS (`~/projects/`) |
| `docker compose` not found | Use `docker-compose` (hyphenated) or update Docker Desktop |

---

## Performance comparison: Local Docker vs RunPod

| Metric | Local (RTX 5070 Ti 16GB) | RunPod (A100 80GB) |
|--------|--------------------------|-------------------|
| VRAM | 16 GB | 80 GB |
| batch_size | 1 | 8 |
| FlashAttention-2 | ✓ (via Docker image) | ✓ (pre-installed) |
| Time per step | ~25–35s | ~1.5–2s |
| Time per epoch | ~14–19h | ~2–3h |
| Cost | $0 (electricity) | ~$1.39/hr × 4h = ~$5.60 |
| Best for | Overnight runs, free experiments | Primary thesis model, fast iteration |
