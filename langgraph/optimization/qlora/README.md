# QLoRA Fine-Tuning — Qwen 2.5 7B

Fine-tunes Qwen 2.5 7B (4-bit quantized) with LoRA adapters for crypto trade-direction prediction.

## Structure

```
qlora/
├── train_qlora.py       ← Main training script (self-contained pipeline)
├── setup_ec2.sh         ← One-command GPU-instance setup (EC2 DLAMI; also SageMaker)
├── run_cloud.sh         ← ONE cloud training run on the fixed split
├── run_local.sh         ← ONE local training run (RTX 5070 Ti), optional
├── plot_overfitting.py  ← Overfitting dashboard from a result JSON
├── EC2_GUIDE.md         ← Step-by-step EC2 walkthrough + cost analysis
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

## Quick Start (EC2 g6e.xlarge — recommended)

```bash
# On the EC2 instance (Deep Learning AMI) terminal:
git clone git@github.com:luisaga215/trading_management.git
cd trading_management/langgraph
bash optimization/qlora/setup_ec2.sh

# REQUIRED before training/eval — without candles, win_rate/PF/Sharpe come out 0:
dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles

# Then (single run, nohup survives disconnects):
source .venv/bin/activate
nohup bash optimization/qlora/run_cloud.sh > optimization/qlora/logs/run_cloud.log 2>&1 &
```

See [EC2_GUIDE.md](EC2_GUIDE.md) for the full walkthrough + cost analysis (the same
`setup_ec2.sh` also works on a SageMaker Studio terminal as a fallback).

## Quick Start (Local — Windows RTX 5070 Ti)

```powershell
cd langgraph
.\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py --epochs 1 --tag qlora_local
```

## What the script does

`train_qlora.py` runs a 5-step pipeline:

1. **prepare_data()** — exports 56K labeled samples into chat-format JSONL (strict temporal split + embargo, 70/15/15)
2. **load_model()** — loads Qwen 2.5 7B in 4-bit + applies LoRA adapters (~40M trainable params, 0.6% of total)
3. **train()** — SFT with TRL's SFTTrainer, cosine LR, checkpoints every 250 steps, **early stopping** (patience 3), keeps best by eval_loss
4. **save_loss_curve()** — dumps `trainer.state.log_history` → `results/<tag>_loss_curve.json` + PNG (train vs eval loss)
5. **save_model()** — saves LoRA adapters (~80MB) + merged GGUF (Q4_K_M, ~4GB) for Ollama
6. **evaluate()** — greedy decoding on the FULL test split, trade simulation, full metrics, **plus** train/val/test accuracy + `gap` and a heuristic baseline

## Overfitting diagnostics (new)

The audit found the old runs could not answer "is it overfitted?". `evaluate()` now emits:
- `overfitting`: `train_acc`, `val_acc`, `test_acc`, `gap = train_acc - test_acc` (the direct memorization signal)
- `baseline_metrics`: a heatmap/MACD heuristic on the same test set (majority-class is only ~51%, so this is the honest contrast)
- `<tag>_loss_curve.{json,png}`: train vs eval loss per step (visual divergence check)

Read thresholds in `docs/GUIA_IMPLEMENTACION_FIX_QLORA.md` Apéndice A.

## Recommended config

A single config (best from the audit), trained on the fixed split:

| LR | Rank | Alpha | Epochs | Batch | Grad accum |
|-----|------|-------|--------|-------|------------|
| 2e-5 | 16 | 32 | 3 | 2 (cloud) / 1 (local) | 8 (cloud) / 16 (local) |

`optimization/configs/qlora.yaml` keeps the other combos as a reference only — they
are **not** swept anymore.

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
--batch-size INT        Per-device batch size (default: 1 local, use 2 on SageMaker)
--grad-accum INT        Gradient accumulation steps (default: 16)
--diagnostic-samples N  Train/val subset for the overfitting gap (default 500; 0 disables)
--tag STR               Run tag — names result/loss-curve files and the model id
--max-steps INT         Cap training steps (for smoke tests)
--max-eval INT          Cap evaluation samples (for quick testing)
--output-dir PATH       Where adapters/checkpoints are written
--resume                Resume from latest checkpoint in output_dir
--model STR             Override base model name/path
--config PATH           Load config from YAML file
```

## Timing

| Environment | Per epoch | 3 epochs | Cost |
|-------------|-----------|----------|------|
| Local (RTX 5070 Ti 16GB) | ~15-19h | ~2 days | "free" |
| SageMaker ml.g6e.xlarge (L40S 48GB) | ~1-2h | ~3-6h | ~$6-12 |
