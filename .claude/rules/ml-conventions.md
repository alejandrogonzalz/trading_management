# ML Conventions — Trading Prediction Project

## Data Handling
- Temporal split 70/15/15 — NEVER shuffle financial time series
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
- Backtest results: `langgraph/backtest/data/results/{tag}.json`
- Model files: torch.save() for LSTM (.pt), pickle for sklearn (.pkl), .save_model() for XGBoost

## Current Best Models (Avance 4, 2026-05-31)
- LSTM: 81.5% test acc (hidden=32, layers=3, seq_len=5, dropout=0.1, lr=0.001)
- SVM (RBF): 70.8% val acc
- XGBoost: 66.6% test acc (tuned with L1/L2 regularization)
