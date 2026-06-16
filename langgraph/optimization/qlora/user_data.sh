#!/bin/bash
# ==============================================================================
# user_data.sh — GPU instance bootstrap for QLoRA training on RunPod/EC2.
#
# Takes a BARE GPU Linux instance (or the unsloth/unsloth:latest image) to
# "training running" with health checks at each step.
#
# Steps:
#   1. Health checks   — OS, GPU, disk, internet
#   2. AWS credentials — REQUIRED via env vars; verified with STS
#   3. AWS CLI v2      — installed if missing
#   4. Base tooling    — git, python3, pip, dvc[s3]
#   5. Repo            — clone at $WORKDIR
#   6. Data            — dvc pull candles + dataset
#   7. Launch          — run_cloud.sh with configured flags
#
# ------------------------------------------------------------------------------
# REQUIRED env vars (export before running):
#
#   export AWS_ACCESS_KEY_ID=AKIA...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_DEFAULT_REGION=us-east-1
#
# OPTIONAL overrides:
#   export REPO_URL=https://github.com/alejandrogonzalz/trading_management.git
#   export REPO_BRANCH=experiment/no-drawdown-filter
#   export WORKDIR=/workspace/work
#   export TAG=qlora_v2_atr          # run tag (names results + model dir)
#   export EPOCHS=1                   # training epochs
#   export USE_ATR_TP_SL=1            # 1 = forward-looking TP/SL (default)
#   export AUTO_LAUNCH=1              # 0 = set up only, don't train
#   export EXTRA_FLAGS=""             # additional flags for run_cloud.sh
#
# Usage:
#   # On a running RunPod/EC2 box:
#   AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... bash user_data.sh
#
#   # With nohup (recommended — survives disconnect):
#   AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... \
#     nohup bash user_data.sh > /workspace/bootstrap.log 2>&1 & echo "PID: $!"
# ==============================================================================

set -uo pipefail

# ----------------------------------------------------------------- config
REPO_URL="${REPO_URL:-https://github.com/alejandrogonzalz/trading_management.git}"
REPO_BRANCH="${REPO_BRANCH:-experiment/no-drawdown-filter}"
WORKDIR="${WORKDIR:-/workspace/work}"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_DEFAULT_REGION
TAG="${TAG:-qlora_v2_atr}"
EPOCHS="${EPOCHS:-1}"
USE_ATR_TP_SL="${USE_ATR_TP_SL:-1}"
AUTO_LAUNCH="${AUTO_LAUNCH:-1}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

LOG="/var/log/user_data.log"
touch "$LOG" 2>/dev/null || LOG="$HOME/user_data.log"
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "============================================================"; echo "  $*"; echo "============================================================"; }
ok()   { echo "  OK   — $*"; }
warn() { echo "  WARN — $*"; }
fail() { echo; echo "  FAIL — $*" >&2; exit 1; }

SUDO=""
if [ "$(id -u)" -ne 0 ]; then command -v sudo >/dev/null 2>&1 && SUDO="sudo" || true; fi

pkg_install() {
    if command -v apt-get >/dev/null 2>&1; then
        $SUDO apt-get update -qq && $SUDO apt-get install -y -qq "$@"
    elif command -v dnf >/dev/null 2>&1; then
        $SUDO dnf install -y -q "$@"
    elif command -v yum >/dev/null 2>&1; then
        $SUDO yum install -y -q "$@"
    else
        warn "no package manager — assuming $* already present"
    fi
}

echo "user_data.sh starting at $(date)"
echo "  TAG=$TAG  EPOCHS=$EPOCHS  USE_ATR_TP_SL=$USE_ATR_TP_SL  BRANCH=$REPO_BRANCH"

# ----------------------------------------------------------------- 1. health checks
step "[1/7] Health checks"
[ "$(uname -s)" = "Linux" ] || fail "this script targets Linux GPU instances"
ok "OS: $(uname -srm)"

if command -v nvidia-smi >/dev/null 2>&1; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)
    ok "GPU: $GPU_INFO"
    # Check no zombie processes holding VRAM
    VRAM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 | tr -d ' ')
    [ "${VRAM_USED:-0}" -gt 5000 ] && warn "GPU already using ${VRAM_USED}MB VRAM — zombie process? Check: nvidia-smi"
else
    fail "nvidia-smi not found — needs a GPU instance with NVIDIA drivers"
fi

AVAIL_GB=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
[ "${AVAIL_GB:-0}" -ge 30 ] && ok "disk: ${AVAIL_GB}GB free" || warn "only ${AVAIL_GB}GB free — need ~30GB for model + candles + checkpoints"

curl -fsS --max-time 10 https://api.github.com >/dev/null 2>&1 && ok "internet reachable" || fail "no internet"

# ----------------------------------------------------------------- 2. AWS credentials
step "[2/7] AWS credentials"
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    fail "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set. Export them first."
fi
ok "credentials present (region=$AWS_DEFAULT_REGION)"

# ----------------------------------------------------------------- 3. AWS CLI
step "[3/7] AWS CLI"
if command -v aws >/dev/null 2>&1; then
    ok "already installed: $(aws --version 2>&1 | awk '{print $1}')"
else
    echo "  installing AWS CLI v2..."
    command -v unzip >/dev/null 2>&1 || pkg_install unzip
    ARCH="$(uname -m)"; [ "$ARCH" = "arm64" ] && ARCH="aarch64"
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${ARCH}.zip" -o /tmp/awscliv2.zip || fail "download failed"
    (cd /tmp && unzip -q -o awscliv2.zip && $SUDO ./aws/install --update) || fail "install failed"
    ok "installed: $(aws --version 2>&1 | awk '{print $1}')"
fi
aws sts get-caller-identity >/dev/null 2>&1 || fail "credentials invalid — check key/secret/region"
ok "verified: $(aws sts get-caller-identity --query Arn --output text 2>/dev/null)"

# ----------------------------------------------------------------- 4. base tooling
step "[4/7] Base tooling"
command -v git >/dev/null 2>&1 || pkg_install git
command -v python3 >/dev/null 2>&1 || pkg_install python3
command -v pip3 >/dev/null 2>&1 || pkg_install python3-pip
ok "git=$(git --version 2>&1 | awk '{print $3}')  python=$(python3 --version 2>&1 | awk '{print $2}')"

# DVC for S3 data pulls
python3 -c "import dvc" 2>/dev/null || pip3 install --quiet "dvc[s3]" || fail "dvc[s3] install failed"
ok "dvc=$(python3 -c 'import dvc; print(dvc.__version__)' 2>/dev/null)"

# ----------------------------------------------------------------- 5. repo
step "[5/7] Repository"
mkdir -p "$WORKDIR"
REPO_DIR="$WORKDIR/trading_management"
if [ -d "$REPO_DIR/.git" ]; then
    echo "  repo exists — pulling latest..."
    git -C "$REPO_DIR" fetch --quiet origin "$REPO_BRANCH"
    git -C "$REPO_DIR" checkout --quiet "$REPO_BRANCH"
    git -C "$REPO_DIR" pull --quiet
else
    git clone --quiet --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR" || fail "git clone failed"
fi
LANGGRAPH="$REPO_DIR/langgraph"
[ -d "$LANGGRAPH" ] || fail "langgraph/ not found — wrong branch?"
cd "$LANGGRAPH"
ok "$(git -C "$REPO_DIR" rev-parse --short HEAD) on $REPO_BRANCH"

# ----------------------------------------------------------------- 6. data
step "[6/7] DVC pull (candles + dataset)"
cd "$REPO_DIR"
dvc pull langgraph/backtest/data/candles || fail "dvc pull candles failed — check AWS creds"
N_CANDLES=$(ls langgraph/backtest/data/candles/*_1h.json 2>/dev/null | wc -l | tr -d ' ')
ok "candles: $N_CANDLES symbols"
[ "$N_CANDLES" -ge 10 ] || fail "only $N_CANDLES candle files — expected 12"

dvc pull langgraph/backtest/data/labeled/dataset.jsonl || fail "dvc pull dataset failed"
N_SAMPLES=$(wc -l < langgraph/backtest/data/labeled/dataset.jsonl | tr -d ' ')
ok "dataset: $N_SAMPLES samples"
[ "$N_SAMPLES" -ge 50000 ] || warn "expected ~56K samples, got $N_SAMPLES"

# ----------------------------------------------------------------- 7. smoke test
step "[7/8] Smoke test (3 steps — catches OOM/import errors in ~1 min)"
cd "$LANGGRAPH"

SMOKE_FLAGS="--max-steps 3 --max-eval 4 --no-gguf --tag ${TAG}_smoke --epochs 1"
[ "$USE_ATR_TP_SL" = "1" ] && SMOKE_FLAGS="$SMOKE_FLAGS --use-atr-tp-sl"

python3 optimization/qlora/train_qlora.py $SMOKE_FLAGS \
    || fail "Smoke test failed — likely OOM or missing dependency. Check nvidia-smi."
ok "smoke test passed — training loop is healthy"

# ----------------------------------------------------------------- 8. launch
step "[8/8] Full training"

CLOUD_FLAGS="--tag $TAG --epochs $EPOCHS"
[ "$USE_ATR_TP_SL" = "1" ] && CLOUD_FLAGS="$CLOUD_FLAGS --use-atr-tp-sl"
[ -n "$EXTRA_FLAGS" ] && CLOUD_FLAGS="$CLOUD_FLAGS $EXTRA_FLAGS"

if [ "$AUTO_LAUNCH" != "1" ]; then
    echo "  AUTO_LAUNCH=0 — setup complete. Run manually:"
    echo "    cd $LANGGRAPH"
    echo "    MAX_EVAL=\"\" bash optimization/qlora/run_cloud.sh $CLOUD_FLAGS"
    exit 0
fi

mkdir -p optimization/qlora/logs
RUN_LOG="optimization/qlora/logs/${TAG}.log"
echo "  Command: MAX_EVAL=\"\" bash optimization/qlora/run_cloud.sh $CLOUD_FLAGS"
echo "  Log: $RUN_LOG"
echo

MAX_EVAL="" bash optimization/qlora/run_cloud.sh $CLOUD_FLAGS 2>&1 | tee "$RUN_LOG"
EXIT_CODE=$?

echo
if [ $EXIT_CODE -eq 0 ]; then
    ok "Training complete! Result: optimization/qlora/results/$TAG/result.json"
else
    fail "Training failed (exit $EXIT_CODE) — check $RUN_LOG"
fi
