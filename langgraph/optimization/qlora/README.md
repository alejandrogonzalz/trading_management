# QLoRA Fine-Tuning — Qwen 2.5 7B

Fine-tunes Qwen 2.5 7B (4-bit quantized) with LoRA adapters for crypto trade-direction prediction.

## Structure

```
qlora/
├── train_qlora.py       ← Main training script (self-contained pipeline)
├── sagemaker_setup.sh   ← One-command SageMaker environment setup
├── run_qlora_search.sh  ← Runs all 5 configs sequentially with logging
├── SAGEMAKER_GUIDE.md   ← Step-by-step SageMaker console walkthrough
├── README.md            ← This file
├── results/             ← Output JSONs from each config (gitignored weights)
└── logs/                ← Training logs per config

# Config lives with the other model configs:
optimization/configs/qlora.yaml  ← Hyperparameter search space + recommended combos
```

## Quick Start (SageMaker)

```bash
# On the SageMaker instance terminal:
git clone https://github.com/alejandrogonzalz/trading_management.git
cd trading_management/langgraph
bash optimization/qlora/sagemaker_setup.sh

# Then:
tmux new -s qlora
source .venv/bin/activate
bash optimization/qlora/run_qlora_search.sh
```

See [SAGEMAKER_GUIDE.md](SAGEMAKER_GUIDE.md) for the full walkthrough.

## Quick Start (Local — Windows RTX 5070 Ti)

```powershell
cd langgraph
.\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py --epochs 1
```

## What the script does

`train_qlora.py` runs a 5-step pipeline:

1. **prepare_data()** — exports 56K labeled samples into chat-format JSONL (train/val/test, 70/15/15 temporal split)
2. **load_model()** — loads Qwen 2.5 7B in 4-bit + applies LoRA adapters (~40M trainable params, 0.6% of total)
3. **train()** — SFT with TRL's SFTTrainer, cosine LR, checkpoints every 250 steps, keeps best by eval_loss
4. **save_model()** — saves LoRA adapters (~80MB) + merged GGUF (Q4_K_M, ~4GB) for Ollama
5. **evaluate()** — greedy decoding on 8,425 test samples, trade simulation, full metrics

## Hyperparameter Configs

From `qlora.yaml` — the 5 recommended combos:

| Config | LR | Rank | Alpha | Epochs | Rationale |
|--------|-----|------|-------|--------|-----------|
| 1 (default) | 2e-5 | 16 | 32 | 3 | Literature default, safe bet |
| 2 | 5e-5 | 8 | 16 | 3 | Aggressive LR, tests if task is "easy" |
| 3 | 1e-5 | 32 | 64 | 3 | Conservative, max capacity |
| 4 | 1e-4 | 16 | 32 | 1 | Fast learning, single epoch |
| 5 | 5e-5 | 16 | 32 | 3 | Mid-range LR, standard rank |

## Output

Each run produces:
- `results/qlora_optimization.json` — full metrics + per-sample predictions for paired statistical tests
- `backtest/data/models/qlora_qwen25_7b/` — LoRA adapters + GGUF model file

## CLI Flags

```
--lr FLOAT         Learning rate (default: 2e-5)
--rank INT         LoRA rank (default: 16)
--alpha INT        LoRA alpha (default: 32)
--epochs INT       Training epochs (default: 3)
--batch-size INT   Per-device batch size (default: 1 local, use 8 on SageMaker)
--max-steps INT    Cap training steps (for smoke tests)
--max-eval INT     Cap evaluation samples (for quick testing)
--resume           Resume from latest checkpoint in output_dir
--model STR        Override base model name/path
--config PATH      Load config from YAML file
```

## Timing

| Environment | Per epoch | 3 epochs | Cost |
|-------------|-----------|----------|------|
| Local (RTX 5070 Ti 16GB) | ~15-19h | ~2 days | "free" |
| SageMaker ml.g6e.xlarge (L40S 48GB) | ~1-2h | ~3-6h | ~$6-12 |
| Full search (5 configs × 3 epochs) | — | ~15-30h | ~$30-60 |
