# QLoRA Fine-Tuning — Qwen 2.5 7B

Fine-tunes Qwen 2.5 7B (4-bit quantized) with LoRA adapters for crypto trade-direction prediction.

## Structure

```
qlora/
├── train_qlora.py       ← Main training script (self-contained pipeline)
├── setup_unsloth_pod.sh ← RunPod setup (validates stack, installs project deps)
├── run_cloud.sh         ← ONE cloud training run on the fixed split
├── run_local.sh         ← ONE local training run (RTX 5070 Ti), optional
├── run_local.ps1        ← PowerShell wrapper for Windows local training
├── setup_ec2.sh         ← Legacy: EC2 DLAMI from-scratch setup (not the primary path)
├── plot_overfitting.py  ← Overfitting dashboard from a result JSON
├── RUNPOD_GUIDE.md      ← Full RunPod walkthrough (primary cloud path)
├── LOCAL_GUIDE.md       ← Local Windows training guide (RTX 5070 Ti)
├── README.md            ← This file
├── results/             ← Output JSONs (+ archive/ for the invalid old sweep)
└── logs/                ← Training logs per run

# Config lives with the other model configs:
optimization/configs/qlora.yaml  ← Hyperparameter reference (NOT swept anymore)
```

> **No config sweep.** The thesis needs one defensible model, not five. The old
> 3-/5-config sweep scripts were removed; their results are archived under
> `results/archive/` and must not be cited (trained on a contaminated split with
> partial single-symbol eval). See `docs/AUDITORIA_QLORA_LEAKAGE_OVERFITTING.md`.

## Quick Start (RunPod A100 80GB — recommended)

```bash
# 1. Clone and set up (no venv needed — /opt/venv is pre-activated in the image)
cd /workspace/work
git clone https://github.com/luisaga215/trading_management.git
cd trading_management/langgraph
bash optimization/qlora/setup_unsloth_pod.sh

# 2. Configure AWS + pull data (REQUIRED for financial metrics)
aws configure
dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles

# 3. Smoke test (3 steps, confirms FA2 + dataset + pipeline work)
python3 optimization/qlora/train_qlora.py \
  --max-steps 3 --max-eval 10 --eval-batch-size 1 \
  --diagnostic-samples 0 --batch-size 4 --tag smoke_full

# 4. Full training run (~4h, ~$5 on A100 @ $1.39/hr)
nohup bash optimization/qlora/run_cloud.sh \
  > optimization/qlora/logs/run_cloud.log 2>&1 &
tail -f optimization/qlora/logs/run_cloud.log
```

See [RUNPOD_GUIDE.md](RUNPOD_GUIDE.md) for the full walkthrough including monitoring
commands, ML grid search in parallel, post-training backup, and troubleshooting.

## Quick Start (Local — Windows RTX 5070 Ti)

```powershell
cd langgraph
.\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py --epochs 1 --tag qlora_local
```

See [LOCAL_GUIDE.md](LOCAL_GUIDE.md) for the full walkthrough.

## What the script does

`train_qlora.py` runs a 5-step pipeline:

1. **prepare_data()** — exports 56K labeled samples into chat-format JSONL (strict temporal split + embargo, 70/15/15)
2. **load_model()** — loads Qwen 2.5 7B in 4-bit + applies LoRA adapters (~40M trainable params, 0.6% of total)
3. **train()** — SFT with TRL's SFTTrainer, cosine LR, checkpoints every 250 steps, **early stopping** (patience 3), keeps best by eval_loss
4. **save_loss_curve()** — dumps `trainer.state.log_history` → `results/<tag>_loss_curve.json` + PNG (train vs eval loss)
5. **save_model()** — saves LoRA adapters (~80MB) + merged GGUF (Q4_K_M, ~4GB) for Ollama
6. **evaluate()** — greedy decoding on the test split (strided), trade simulation, full metrics, **plus** train/val/test accuracy + `gap` and a heuristic baseline

## Overfitting diagnostics

`evaluate()` emits:
- `overfitting`: `train_acc`, `val_acc`, `test_acc`, `gap = train_acc - test_acc`
- `baseline_metrics`: a heatmap/MACD heuristic on the same test set
- `<tag>_loss_curve.{json,png}`: train vs eval loss per step

## Recommended config

A single config trained on the fixed split:

| LR | Rank | Alpha | Epochs | Batch | Grad accum |
|-----|------|-------|--------|-------|------------|
| 2e-5 | 16 | 32 | 2 (cloud) | 8 A100+FA2 / 2 L40S | 2 / 8 |

## Output

Each run produces:
- `results/qlora_<tag>.json` — full metrics + per-sample predictions for paired statistical tests
- `results/qlora_optimization.json` — canonical copy (what `compare-stats` reads by default)
- `results/<tag>_loss_curve.{json,png}` — loss history + chart
- `backtest/data/models/<tag>/` — LoRA adapters + GGUF model file

## CLI Flags

```
--lr FLOAT              Learning rate (default: 2e-5)
--rank INT              LoRA rank (default: 16)
--alpha INT             LoRA alpha (default: 32)
--epochs INT            Training epochs (default: 3)
--batch-size INT        Per-device batch size (default: 1 local, 2-8 cloud)
--grad-accum INT        Gradient accumulation steps (default: 16)
--eval-batch-size INT   Prompts per generate() call during eval (default: 1)
--diagnostic-samples N  Train/val subset for the overfitting gap (default 500; 0 disables)
--tag STR               Run tag — names result/loss-curve files and the model id
--max-steps INT         Cap training steps (for smoke tests)
--max-eval INT          Cap evaluation samples (for smoke tests only)
--output-dir PATH       Where adapters/checkpoints are written
--resume                Resume from latest checkpoint in output_dir
--model STR             Override base model name/path
--config PATH           Load config from YAML file
```

## Timing

| Environment | Per epoch | 2 epochs | Cost |
|-------------|-----------|----------|------|
| RunPod A100 80GB (FA2, batch=8) | ~1.5–2h | **~3–3.5h** | **~$5** |
| RunPod L40S 48GB (FA2, batch=4) | ~3–4h | ~6–7h | ~$6 |
| Local RTX 5070 Ti (Triton, batch=1) | ~15–19h | N/A (1 epoch only) | $0 |
