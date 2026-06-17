#!/bin/bash
# ==============================================================================
# user_data.sh — GPU instance bootstrap for QLoRA training.
#
# A cloud-init / EC2 user-data / SageMaker lifecycle-config style script that takes
# a BARE GPU Linux instance to "training running" with health checks at each step:
#
#   1.  Health checks   — OS, GPU (nvidia-smi), disk, internet
#   2.  AWS credentials — REQUIRED via env vars (see below); verified with STS
#   3.  AWS CLI v2      — installed if missing
#   4.  Base tooling    — git, curl, unzip, python3/pip, Node.js
#   5.  Claude Code     — npm i -g @anthropic-ai/claude-code (for interactive debug)
#   6.  Repo            — clone/update at $WORKDIR
#   7.  Python stack    — dvc[s3] importable, unsloth importable, CUDA visible to torch
#   8.  Data            — dvc pull candles.dvc + dataset.jsonl.dvc (repo root)
#   9.  Smoke test      — 3-step / 10-eval-sample dry run of train_qlora.py
#   10. Launch          — run_cloud.sh (the real training + DVC/git save)
#
# ------------------------------------------------------------------------------
# REQUIRED: export AWS credentials in the environment BEFORE running this script.
# (DVC's S3 remote and the AWS CLI both read them from the environment — never
#  hardcode secrets in this file.)
#
#   export AWS_ACCESS_KEY_ID=AKIA...
#   export AWS_SECRET_ACCESS_KEY=...
#   export AWS_DEFAULT_REGION=us-east-1
#
# RECOMMENDED (so Claude Code works non-interactively for debugging):
#   export ANTHROPIC_API_KEY=sk-ant-...
#
# OPTIONAL overrides (have sensible defaults):
#   export REPO_URL=https://github.com/luisaga215/trading_management.git
#   export REPO_BRANCH=dev
#   export WORKDIR=/workspace/work            # where the repo is cloned
#   export AUTO_LAUNCH=1                       # 0 = set everything up but don't train
#   export SMOKE_ONLY=0                        # 1 = stop after the smoke test
#   export TRAIN_TAG=qlora_cloud               # names the run_cloud.sh result/model
#   export TRAIN_FLAGS=""                      # extra flags forwarded to run_cloud.sh
#
# Usage:
#   As EC2 user-data: paste this file (cloud-init runs it as root at first boot;
#     output lands in /var/log/cloud-init-output.log). Set the env vars via an
#     instance profile / SSM / a prepended `export` block.
#   Manually on a running box:
#     AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... AWS_DEFAULT_REGION=us-east-1 \
#     ANTHROPIC_API_KEY=... bash user_data.sh
# ==============================================================================

set -uo pipefail

# Absolute path to this script's directory (optimization/qlora/), independent of
# whatever the CWD happens to be after the repo clone/cd dance below.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ----------------------------------------------------------------- config + logging
REPO_URL="${REPO_URL:-https://github.com/luisaga215/trading_management.git}"
REPO_BRANCH="${REPO_BRANCH:-dev}"
WORKDIR="${WORKDIR:-/workspace/work}"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_DEFAULT_REGION
AUTO_LAUNCH="${AUTO_LAUNCH:-1}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
TRAIN_TAG="${TRAIN_TAG:-qlora_cloud}"
TRAIN_FLAGS="${TRAIN_FLAGS:-}"
LOG="/var/log/user_data_experiment.log"
touch "$LOG" 2>/dev/null || LOG="$HOME/user_data_experiment.log"
# Mirror everything to the log file as well as the console.
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "============================================================"; echo "  $*"; echo "============================================================"; }
ok()   { echo "  OK   — $*"; }
warn() { echo "  WARN — $*"; }
fail() { echo; echo "  FAIL — $*" >&2; exit 1; }

# sudo only when not already root AND passwordless sudo actually works (containers
# commonly ship a sudo binary that requires a password nobody has — `command -v sudo`
# alone is not a reliable signal, so probe it with `sudo -n`).
SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1 && sudo -n true 2>/dev/null; then
        SUDO="sudo"
    else
        warn "not root and no passwordless sudo — package installs may fail; falling back to user-local installs where possible"
    fi
fi

# Package-manager abstraction (Ubuntu/Debian apt vs Amazon Linux/RHEL yum/dnf).
if command -v apt-get >/dev/null 2>&1; then PKG="apt"; elif command -v dnf >/dev/null 2>&1; then PKG="dnf"; elif command -v yum >/dev/null 2>&1; then PKG="yum"; else PKG=""; fi
pkg_install() {
    case "$PKG" in
        apt) $SUDO apt-get update -qq && $SUDO apt-get install -y -qq "$@" ;;
        dnf) $SUDO dnf install -y -q "$@" ;;
        yum) $SUDO yum install -y -q "$@" ;;
        *)   warn "no known package manager — assuming $* already present" ;;
    esac
}

echo "user_data.sh starting at $(date) — log: $LOG"

# ----------------------------------------------------------------- 1. health checks
step "[1/10] Health checks"
[ "$(uname -s)" = "Linux" ] || fail "this bootstrap targets Linux GPU instances"
ok "OS: $(uname -srm)"

if command -v nvidia-smi >/dev/null 2>&1; then
    ok "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | head -1)"
else
    fail "nvidia-smi not found — this needs a GPU instance with NVIDIA drivers"
fi

AVAIL_GB=$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')
AVAIL_GB="${AVAIL_GB:-0}"
[ "$AVAIL_GB" -ge 40 ] && ok "disk: ${AVAIL_GB}GB free on /" || warn "only ${AVAIL_GB}GB free on / — model + candles + checkpoints want ~40GB+"

curl -fsS --max-time 10 https://api.github.com >/dev/null 2>&1 && ok "internet reachable" || fail "no internet — cannot clone/install"

# ----------------------------------------------------------------- 2. AWS credentials
step "[2/10] AWS credentials (from env vars)"
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    fail "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set. Export them first:
    export AWS_ACCESS_KEY_ID=AKIA...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION
  (DVC's S3 remote reads these from the environment.)"
fi
ok "credentials present in env (region=$AWS_DEFAULT_REGION)"

# ----------------------------------------------------------------- 3. AWS CLI v2
step "[3/10] AWS CLI"
if command -v aws >/dev/null 2>&1; then
    ok "already installed: $(aws --version 2>&1)"
else
    echo "  installing AWS CLI v2..."
    command -v unzip >/dev/null 2>&1 || pkg_install unzip
    ARCH="$(uname -m)"; [ "$ARCH" = "arm64" ] && ARCH="aarch64"
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${ARCH}.zip" -o /tmp/awscliv2.zip || fail "AWS CLI download failed"
    (cd /tmp && unzip -q -o awscliv2.zip) || fail "AWS CLI unzip failed"
    if [ -n "$SUDO" ]; then
        (cd /tmp && $SUDO ./aws/install --update) || fail "AWS CLI install failed"
    else
        # No root available — install directly under $HOME (NOT $HOME/.local:
        # in some images that dir is pre-created root-owned with only specific
        # subdirs like .local/bin made user-writable, so mkdir on a new path
        # under it fails even though $HOME itself is writable).
        mkdir -p "$HOME/aws-cli" "$HOME/bin" || fail "could not create install dirs under \$HOME ($HOME)"
        (cd /tmp && ./aws/install --install-dir "$HOME/aws-cli" --bin-dir "$HOME/bin" --update) \
            || fail "AWS CLI user-local install failed"
        export PATH="$HOME/bin:$PATH"
    fi
    command -v aws >/dev/null 2>&1 || fail "aws still not on PATH after install"
    ok "installed: $(aws --version 2>&1)"
fi
# Verify the credentials actually work BEFORE we depend on them for dvc pull.
aws sts get-caller-identity >/dev/null 2>&1 || fail "aws sts get-caller-identity failed — bad/expired credentials or wrong region"
ok "credentials verified: $(aws sts get-caller-identity --query Arn --output text 2>/dev/null)"

# ----------------------------------------------------------------- 4. base tooling
step "[4/10] Base tooling (git, python3, node)"
command -v git    >/dev/null 2>&1 || pkg_install git
command -v curl   >/dev/null 2>&1 || pkg_install curl
command -v python3 >/dev/null 2>&1 || pkg_install python3
command -v pip3   >/dev/null 2>&1 || pkg_install python3-pip
ok "git=$(git --version 2>&1 | awk '{print $3}')  python3=$(python3 --version 2>&1 | awk '{print $2}')"

if command -v node >/dev/null 2>&1; then
    ok "node: $(node --version)"
else
    echo "  installing Node.js 20 (for Claude Code)..."
    if [ "$PKG" = "apt" ]; then
        curl -fsSL https://deb.nodesource.com/setup_20.x | $SUDO -E bash - && pkg_install nodejs
    else
        curl -fsSL https://rpm.nodesource.com/setup_20.x | $SUDO -E bash - && pkg_install nodejs
    fi
    command -v node >/dev/null 2>&1 && ok "node: $(node --version)" || warn "Node install failed — Claude Code step will be skipped"
fi

# ----------------------------------------------------------------- 5. Claude Code
step "[5/10] Claude Code CLI"
if command -v claude >/dev/null 2>&1; then
    ok "already installed: $(claude --version 2>&1 | head -1)"
elif command -v npm >/dev/null 2>&1; then
    # Try without sudo first — npm's global prefix is often a user-writable
    # directory (e.g. nvm/~/.local) even when system-wide sudo isn't available.
    if npm install -g @anthropic-ai/claude-code >/dev/null 2>&1 \
        || { [ -n "$SUDO" ] && $SUDO npm install -g @anthropic-ai/claude-code >/dev/null 2>&1; }; then
        ok "installed: $(claude --version 2>&1 | head -1)"
    else
        warn "Claude Code install failed (non-fatal — the experiment script runs without it)"
    fi
else
    warn "npm unavailable — skipping Claude Code (experiment still runs)"
fi
[ -n "${ANTHROPIC_API_KEY:-}" ] && ok "ANTHROPIC_API_KEY set (claude works non-interactively)" \
    || warn "ANTHROPIC_API_KEY not set — run 'claude' interactively to log in if you want agent debugging"

# ----------------------------------------------------------------- 6. repo
step "[6/10] Repository"
mkdir -p "$WORKDIR" || fail "cannot create $WORKDIR"
REPO_DIR="$WORKDIR/trading_management"
if [ -d "$REPO_DIR/.git" ]; then
    echo "  repo present — fetching $REPO_BRANCH..."
    git -C "$REPO_DIR" fetch --quiet origin "$REPO_BRANCH" && git -C "$REPO_DIR" checkout --quiet "$REPO_BRANCH" && git -C "$REPO_DIR" pull --quiet || warn "git update had issues; continuing with current checkout"
else
    git clone --quiet --branch "$REPO_BRANCH" "$REPO_URL" "$REPO_DIR" || fail "git clone failed ($REPO_URL @ $REPO_BRANCH)"
fi
LANGGRAPH="$REPO_DIR/langgraph"
[ -d "$LANGGRAPH" ] || fail "langgraph/ not found in repo — wrong REPO_URL/branch?"
cd "$LANGGRAPH"
ok "repo ready: $REPO_DIR @ $(git -C "$REPO_DIR" rev-parse --short HEAD) ($REPO_BRANCH)"

# ----------------------------------------------------------------- 7. python stack
step "[7/10] Python stack"
python3 -c "import dvc" 2>/dev/null || pip3 install --quiet "dvc[s3]" || fail "failed to install dvc[s3]"
python3 -c "import dvc" 2>/dev/null && ok "dvc: $(python3 -c 'import dvc; print(dvc.__version__)' 2>/dev/null)" || fail "dvc not importable after install attempt"

python3 -c "import unsloth" 2>/dev/null \
    && ok "unsloth importable" \
    || fail "unsloth not importable — this script must run on a GPU image with the QLoRA training stack pre-installed (e.g. unsloth/unsloth:latest)"

python3 -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null \
    && ok "torch.cuda.is_available(): True ($(python3 -c 'import torch; print(torch.cuda.get_device_name(0))' 2>/dev/null))" \
    || fail "torch.cuda.is_available() is False — no GPU visible to PyTorch"

# ----------------------------------------------------------------- 8. data (dvc pull)
step "[8/10] Data — dvc pull (candles + labeled dataset)"
# DVC commands must run from the repo root (the DVC root), not from langgraph/.
cd "$REPO_DIR" || fail "cannot cd to repo root $REPO_DIR"
dvc pull langgraph/backtest/data/candles.dvc || fail "dvc pull candles failed — check AWS creds/region and 'dvc remote list'"
ok "candles pulled"
dvc pull langgraph/backtest/data/labeled/dataset.jsonl.dvc || fail "dvc pull dataset.jsonl failed"
ok "labeled dataset pulled"
cd "$LANGGRAPH" || fail "cannot cd back to $LANGGRAPH"

# ----------------------------------------------------------------- 9. smoke test
step "[9/10] Smoke test (3 steps, 10 eval samples)"
mkdir -p "$SCRIPT_DIR/logs"
SMOKE_LOG="$SCRIPT_DIR/logs/smoke_test.log"

# train_qlora.py always (re)writes results/qlora_optimization.json as the
# "latest canonical" result, even for a 3-step smoke test — back it up so a
# real previous run's numbers aren't clobbered by smoke-test noise.
CANONICAL_RESULT="$SCRIPT_DIR/results/qlora_optimization.json"
CANONICAL_BACKUP="$CANONICAL_RESULT.user_data_bak"
[ -f "$CANONICAL_RESULT" ] && cp "$CANONICAL_RESULT" "$CANONICAL_BACKUP"

python3 "$SCRIPT_DIR/train_qlora.py" \
    --max-steps 3 --max-eval 10 --tag smoke_test \
    --output-dir "backtest/data/models/smoke_test" --no-gguf \
    > "$SMOKE_LOG" 2>&1
SMOKE_STATUS=$?

if [ "$SMOKE_STATUS" -ne 0 ]; then
    echo "  ---- last 30 lines of $SMOKE_LOG ----"
    tail -30 "$SMOKE_LOG"
    fail "smoke test failed (exit $SMOKE_STATUS) — see $SMOKE_LOG"
fi
ok "smoke test passed — training loop, eval, and result serialization all work"

# Clean up smoke-test artifacts so they don't pollute real results/models.
rm -rf "$SCRIPT_DIR/results/smoke_test" "$LANGGRAPH/backtest/data/models/smoke_test"
if [ -f "$CANONICAL_BACKUP" ]; then
    mv "$CANONICAL_BACKUP" "$CANONICAL_RESULT"
else
    rm -f "$CANONICAL_RESULT"
fi
ok "smoke-test artifacts cleaned up"

# ----------------------------------------------------------------- 10. launch
TRAIN_LOG="$SCRIPT_DIR/logs/${TRAIN_TAG}.log"
LAUNCH_CMD="cd $LANGGRAPH && nohup bash optimization/qlora/run_cloud.sh --tag $TRAIN_TAG $TRAIN_FLAGS > $TRAIN_LOG 2>&1 & echo \"PID: \$!\""

if [ "$SMOKE_ONLY" = "1" ]; then
    step "[10/10] Full training launch — SKIPPED (SMOKE_ONLY=1)"
    echo "  Smoke test passed. Launch the full run yourself with:"
    echo "    $LAUNCH_CMD"
    exit 0
fi

if [ "$AUTO_LAUNCH" != "1" ]; then
    step "[10/10] Full training launch — SKIPPED (AUTO_LAUNCH=0)"
    echo "  Everything is set up. Launch the full run yourself with:"
    echo "    $LAUNCH_CMD"
    exit 0
fi

step "[10/10] Full training launch"
echo "  launching: bash optimization/qlora/run_cloud.sh --tag $TRAIN_TAG $TRAIN_FLAGS  (log → $TRAIN_LOG)"
nohup bash "$SCRIPT_DIR/run_cloud.sh" --tag "$TRAIN_TAG" $TRAIN_FLAGS > "$TRAIN_LOG" 2>&1 &
PID=$!
ok "training started (PID $PID)"
echo
echo "  Monitor:   tail -f $TRAIN_LOG"
echo "  GPU:       watch -n 10 nvidia-smi"
echo "  Result:    $SCRIPT_DIR/results/$TRAIN_TAG/result.json (canonical: $SCRIPT_DIR/results/qlora_optimization.json)"
echo "  user_data log: $LOG"
