# Docker Local Training — Steering

## What this is
Docker-based QLoRA fine-tuning on Windows GPU using `docker.io/unsloth/unsloth:latest`.
Same environment as RunPod — FA2 pre-compiled, no platform compatibility issues.

## Key files
| File | Purpose |
|------|---------|
| `langgraph/optimization/qlora/docker-compose.local.yml` | Docker Compose config — simplest launch method |
| `langgraph/optimization/qlora/run_docker_windows.ps1` | PowerShell launcher (more control, flag passing) |
| `langgraph/optimization/qlora/run_docker_local.sh` | Inner script (runs inside container): health checks → smoke test → full training |
| `langgraph/optimization/qlora/setup_docker_local.sh` | Standalone validation script (does NOT train) |
| `langgraph/optimization/qlora/LOCAL_GUIDE.md` | Full step-by-step guide with prerequisites |

## Critical invariants
- DVC pull happens on the HOST (Windows), never inside the container
- The container mounts `langgraph/` as `/workspace` — all paths are relative to that
- `--gpus all` is required or GPU will not be visible inside the container
- Health checks run automatically before training (pass `--skip-checks` to bypass)
- Smoke test (3 steps) runs before the full training to catch OOM early
- Results write to the host filesystem via bind mount — they persist after container exits

## Volume mount structure
```
HOST: C:\Users\alex\projects\trading_management\langgraph\
                    ↓ (bind mount)
CONTAINER: /workspace/
  ├── backtest/data/labeled/dataset.jsonl  ← read by training
  ├── backtest/data/candles/               ← read by evaluation
  ├── backtest/data/models/<tag>/          ← adapters + GGUF written here
  ├── optimization/qlora/results/          ← result JSON written here
  └── optimization/qlora/logs/             ← training logs written here
```

## Common issues
- `nvidia-smi not found` → NVIDIA Container Toolkit not installed on host (WSL2 side)
- `torch.cuda.is_available() = False` → Docker not launched with `--gpus all`, or stale image
- `dataset.jsonl not found` → forgot to `dvc pull` on the host before launching container
- `win_rate = 0` → candle files missing, need `dvc pull backtest/data/candles` on host
- OOM on step 0 → Ollama or Chrome using GPU; close them and retry
- Very slow (>60s/step) → FA2 not available in image; `docker pull` to get latest

## Default config (local, 16GB VRAM)
- lr=5e-5, rank=8, alpha=16, epochs=1, batch_size=1, grad_accum=16
- Produces: `optimization/qlora/results/qlora_qlora_local.json`
- Model: `backtest/data/models/qlora_local/`

## How to run different configs
```bash
# Aggressive (same as cloud config2)
docker compose -f optimization/qlora/docker-compose.local.yml up
# with: QLORA_FLAGS="--tag config2 --lr 0.00005 --rank 8 --epochs 3"

# Conservative (config3)
# QLORA_FLAGS="--tag config3 --lr 0.00001 --rank 32 --alpha 64 --epochs 2"
```
