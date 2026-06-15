# ML Conventions — Trading Prediction Project

## Data Handling
- Temporal split 70/15/15 — NEVER shuffle financial time series
- **"Temporal split" means a strict temporal holdout**: `features._temporal_split`
  sorts ALL samples globally by `timestamp` and applies a time-based **embargo**
  (default 24 bars × base TF) at each val/test boundary to purge the labeler's
  lookahead. **File order ≠ temporal order** — `dataset.jsonl` is grouped by
  symbol, so a positional cut leaks (per-symbol, not by time). Do not reintroduce
  a positional split. (History: `docs/AUDITORIA_QLORA_LEAKAGE_OVERFITTING.md`.)
- Random seed: 42 everywhere (numpy, torch, sklearn, random)
- StandardScaler fit on train only, transform val/test separately
- Dataset path: `langgraph/backtest/data/labeled/dataset.jsonl` (56,161 samples)

## Feature Pipeline
- XGBoost / RF use raw features (tree-based models are scale-invariant)
- SVM / KNN / MLP / LogReg use StandardScaler-transformed features
- LSTM sequences: zero-pad from left, normalize with train mean/std
- Feature vector: 9 per timeframe × n_timeframes + 3 cross-TF = dynamic size

## Training
- Early stopping patience=10 for neural models (LSTM, MLP)
- TimeSeriesSplit for cross-validation (never KFold on time series)
- OMP_NUM_THREADS=1 when mixing XGBoost/sklearn with PyTorch in same process

## Results & Serialization
- Optimization results: `langgraph/optimization/results/{model}_optimization.json`
- QLoRA results: `langgraph/optimization/qlora/results/qlora_<tag>.json` (+ canonical `qlora_optimization.json`)
- Backtest results: `langgraph/backtest/data/results/{tag}.json`
- Model files: torch.save() for LSTM (.pt), pickle for sklearn (.pkl), .save_model() for XGBoost
- QLoRA models: adapters + GGUF in `langgraph/backtest/data/models/<tag>/` (DVC-tracked)

## Current Best Models (2026-06-13)
- ⚠️ **All numbers below predate the leakage fix and must be RE-MEASURED** on the
  strict temporal split before citing. They were trained on a positional
  (per-symbol) split — see `docs/AUDITORIA_QLORA_LEAKAGE_OVERFITTING.md` + Tarea 7
  of `docs/GUIA_IMPLEMENTACION_FIX_QLORA.md`.
- ~~QLoRA fine-tuned (config 1): 92% direction acc~~ — **INVALID** (contaminated
  split + partial single-symbol eval, financial metrics 0). Archived under
  `optimization/qlora/results/archive/`. Re-train one model (`run_cloud.sh`).
- Bagging-LSTM: 83.37% test acc (5 bags, AUC-ROC=0.9157) — best ML model (Avance 5),
  **re-measure** (same `_temporal_split`, same leakage)
- LSTM individual: 81.5% test acc (hidden=32, layers=3, seq_len=5, dropout=0.1, lr=0.001)
- Blending ensemble: 81.89% test acc
- SVM (RBF): 70.8% val acc
- XGBoost: 66.6% test acc (tuned with L1/L2 regularization)

## QLoRA Fine-tuning
- Script: `langgraph/optimization/qlora/train_qlora.py`
- Venv: `langgraph/.venv-finetuning/` (local Windows) or `langgraph/.venv/` (SageMaker)
- Base model: `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
- max_seq_length=1024 (NEVER lower — samples are 877-933 tokens)
- DVC remote: `s3://trading-management-dvc/`
