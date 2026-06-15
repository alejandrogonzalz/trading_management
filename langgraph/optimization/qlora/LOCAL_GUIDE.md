# QLoRA Local Training — Docker (RTX 5070 Ti 16GB)

Runs the same `docker.io/unsloth/unsloth:latest` image used on RunPod, locally on the
RTX 5070 Ti. FA2 is pre-installed in the image — no venv, no manual dependency setup.

**When to use**: free second data point alongside the cloud model. Same fixed temporal
split, 1 epoch, ~14h overnight. Cloud (RunPod) is the primary result.

---

## Prerequisites

| What | Check |
|------|-------|
| Docker Desktop with GPU support | `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu22.04 nvidia-smi` |
| NVIDIA driver ≥ R570 (≥ 572.xx on Windows) | `nvidia-smi` → driver version top row |
| ~30 GB free disk | Model cache ~5GB, checkpoints ~15GB, GGUF ~4GB, data ~2GB |
| GPU free (close Ollama, games, Chrome GPU) | `nvidia-smi` → 0 MiB used |
| AWS CLI configured (for DVC pull) | `aws sts get-caller-identity` |

---

## Step 1 — Pull data (REQUIRED)

Without candles, `win_rate` / `profit_factor` / `Sharpe` come out 0.

```bash
# From langgraph/ (WSL2 or Git Bash)
dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles
python3 -c "print(sum(1 for _ in open('backtest/data/labeled/dataset.jsonl')))"
# Must print: 56161
```

---

## Step 2 — Run (WSL2 recommended)

Launch from `langgraph/`. The script smoke-tests first (3 steps, OOM check), then
starts the full run automatically.

```bash
# WSL2 (best I/O — keep the project in the WSL2 filesystem)
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "$(pwd):/workspace" -w /workspace \
  docker.io/unsloth/unsloth:latest \
  bash optimization/qlora/run_docker_local.sh
```

```powershell
# PowerShell alternative
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 `
  -v "${PWD}:/workspace" -w /workspace `
  docker.io/unsloth/unsloth:latest `
  bash optimization/qlora/run_docker_local.sh
```

**Check the startup banner** — must show `FA [FA2 = True]`. If it shows `FA2 = False`,
the image may be stale; run `docker pull docker.io/unsloth/unsloth:latest`.

### Detached (survives closing the terminal — WSL2 only)

```bash
nohup docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "$(pwd):/workspace" -w /workspace \
  docker.io/unsloth/unsloth:latest \
  bash optimization/qlora/run_docker_local.sh \
  > optimization/qlora/logs/run_local.log 2>&1 &
echo "PID: $!"
```

---

## Step 3 — Monitor (new terminal)

```bash
# Training progress table (refreshes every 30s)
python3 optimization/qlora/monitor_training.py \
  --log optimization/qlora/logs/run_local.log --epochs 1 --watch

# Raw log tail
tail -f optimization/qlora/logs/run_local.log

# GPU stats
nvidia-smi --query-gpu=temperature.gpu,utilization.gpu,memory.used,memory.total \
  --format=csv -l 10
```

### Healthy signals

| Metric | Expected |
|--------|----------|
| s/step | ~25–35s |
| GPU memory | ~10–13 GB / 16 GB |
| Loss (first 100 steps) | 1.5–2.5, decreasing |
| Loss (step 500+) | 0.7–1.2, stable |

---

## Step 4 — If interrupted

Checkpoints save every 250 steps. Resume by appending `--resume` to the docker command:

```bash
docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  -v "$(pwd):/workspace" -w /workspace \
  docker.io/unsloth/unsloth:latest \
  bash optimization/qlora/run_docker_local.sh --resume
```

> If a smoke-test `checkpoint-3` is left over in the output dir, delete it first:
> `rm -rf backtest/data/models/qlora_local/checkpoint-3`

---

## Step 5 — Results

```
results/qlora_local/
├── result.json        ← full metrics + predictions (paired with cloud via sample_keys)
├── loss_curve.json    ← per-step train/eval loss
└── loss_curve.png     ← overfitting chart

backtest/data/models/qlora_local/
├── adapter_model.safetensors
├── gguf_gguf/Qwen2.5-7B-Instruct.Q4_K_M.gguf
└── ...
```

Compare cloud vs local:

```bash
python3 -m cli compare-stats \
  --a optimization/qlora/results/qlora_cloud/result.json \
  --b optimization/qlora/results/qlora_local/result.json
```

---

## Step 6 — Save to DVC + git

Run from `trading_management/` (the DVC root):

```bash
dvc add langgraph/backtest/data/models/qlora_local/
dvc push
git add langgraph/backtest/data/models/qlora_local.dvc \
        langgraph/backtest/data/models/.gitignore \
        langgraph/optimization/qlora/results/qlora_local/
git commit -m "feat(qlora): local training results (RTX 5070 Ti)"
git push
```

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| OOM on step 0 | Close Ollama, Chrome, games — need ~3 GB free headroom |
| OOM on step N | VRAM fragmentation — restart container, `--resume` |
| `FA2 = False` | `docker pull docker.io/unsloth/unsloth:latest` |
| Very slow (>60 s/step) | Something competing for GPU — check Task Manager GPU tab |
| Loss = NaN | LR too high — add `--lr 0.00002` |
| Container exits immediately | Check `docker run` output for CUDA / driver errors |
| Project on Windows FS (`/mnt/c/`) | Move project to WSL2 FS (`~/`) — Docker I/O is very slow across the boundary |
