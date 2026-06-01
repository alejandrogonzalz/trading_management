# Cloud Desktop Setup

## Machine Specs (dev-dsk-alexglz-1d)
- AMD EPYC 7R13, 8 vCPUs
- 15 GB RAM (~12 GB available)
- Amazon Linux 2
- No GPU (CPU-only training)

## Initial Setup (one-time)
```bash
# Clone repo
git clone <repo-url> ~/trading_management
cd ~/trading_management/langgraph

# Install Python 3.11 if not available
sudo amazon-linux-extras install python3.11 2>/dev/null || \
  sudo yum install python311 -y

# Create venv
python3.11 -m venv .venv
source .venv/bin/activate

# Install deps
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn xgboost pandas numpy matplotlib jupyter tqdm pyyaml
```

## Running Optimization
```bash
cd ~/trading_management/langgraph
source .venv/bin/activate

# Use tmux to persist across SSH disconnects
tmux new -s training

# Inside tmux:
OMP_NUM_THREADS=4 python optimization/run_ensembles.py \
  --dataset backtest/data/labeled/dataset.jsonl \
  > logs/ensembles.log 2>&1

# Detach: Ctrl+B then D
# Reattach later: tmux attach -t training
```

## Performance Notes
- 8 vCPUs → set OMP_NUM_THREADS=4 (leave headroom for OS)
- 12 GB RAM is enough for all models (LSTM peak ~2GB, XGBoost ~1GB)
- No sleep risk — Cloud Desktop stays up
- PyTorch CPU-only is fine — LSTM(hidden=32, seq=5) trains in ~30-40s per config
- sklearn n_jobs=-1 will use all 8 vCPUs for RandomizedSearchCV
- tmux prevents job death on SSH disconnect
