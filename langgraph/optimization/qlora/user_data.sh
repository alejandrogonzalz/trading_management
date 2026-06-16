#!/bin/bash
# ==============================================================================
# user_data.sh — GPU instance bootstrap for the no-drawdown-filter experiment.
#
# A cloud-init / EC2 user-data / SageMaker lifecycle-config style script that takes
# a BARE GPU Linux instance to "experiment running" with health checks at each step:
#
#   1. Health checks   — OS, GPU (nvidia-smi), disk, internet
#   2. AWS credentials — REQUIRED via env vars (see below); verified with STS
#   3. AWS CLI v2      — installed if missing
#   4. Base tooling    — git, curl, unzip, python3/pip, Node.js
#   5. Claude Code     — npm i -g @anthropic-ai/claude-code (for interactive debug)
#   6. Repo            — clone/update at $WORKDIR
#   7. Python stack    — delegates to setup_unsloth_pod.sh when present (unsloth
#                        image); otherwise ensures dvc[s3] is available
#   8. Data            — dvc pull CANDLES (required) + dataset.jsonl (baseline)
#   9. Launch          — runs run_experiment_no_drawdown_filter.sh (smoke → full)
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
#   export REPO_BRANCH=experiment/no-drawdown-filter
#   export WORKDIR=/workspace/work            # where the repo is cloned
#   export AUTO_LAUNCH=1                       # 0 = set everything up but don't train
#   export SMOKE_ONLY=0                        # 1 = stop after the 3-step smoke test
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

# ----------------------------------------------------------------- config + logging
REPO_URL="${REPO_URL:-https://github.com/luisaga215/trading_management.git}"
REPO_BRANCH="${REPO_BRANCH:-experiment/no-drawdown-filter}"
WORKDIR="${WORKDIR:-/workspace/work}"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-us-east-1}"
export AWS_DEFAULT_REGION
AUTO_LAUNCH="${AUTO_LAUNCH:-1}"
SMOKE_ONLY="${SMOKE_ONLY:-0}"
LOG="/var/log/user_data_experiment.log"
touch "$LOG" 2>/dev/null || LOG="$HOME/user_data_experiment.log"
# Mirror everything to the log file as well as the console.
exec > >(tee -a "$LOG") 2>&1

step() { echo; echo "============================================================"; echo "  $*"; echo "============================================================"; }
ok()   { echo "  OK   — $*"; }
warn() { echo "  WARN — $*"; }
fail() { echo; echo "  FAIL — $*" >&2; exit 1; }

# sudo only when not already root (containers usually run as root / a sudo-less user).
SUDO=""
if [ "$(id -u)" -ne 0 ]; then command -v sudo >/dev/null 2>&1 && SUDO="sudo" || warn "not root and no sudo — package installs may fail"; fi

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
step "[1/9] Health checks"
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
step "[2/9] AWS credentials (from env vars)"
if [ -z "${AWS_ACCESS_KEY_ID:-}" ] || [ -z "${AWS_SECRET_ACCESS_KEY:-}" ]; then
    fail "AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY not set. Export them first:
    export AWS_ACCESS_KEY_ID=AKIA...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_DEFAULT_REGION=$AWS_DEFAULT_REGION
  (DVC's S3 remote reads these from the environment.)"
fi
ok "credentials present in env (region=$AWS_DEFAULT_REGION)"

# ----------------------------------------------------------------- 3. AWS CLI v2
step "[3/9] AWS CLI"
if command -v aws >/dev/null 2>&1; then
    ok "already installed: $(aws --version 2>&1)"
else
    echo "  installing AWS CLI v2..."
    command -v unzip >/dev/null 2>&1 || pkg_install unzip
    ARCH="$(uname -m)"; [ "$ARCH" = "arm64" ] && ARCH="aarch64"
    curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-${ARCH}.zip" -o /tmp/awscliv2.zip || fail "AWS CLI download failed"
    (cd /tmp && unzip -q -o awscliv2.zip && $SUDO ./aws/install --update) || fail "AWS CLI install failed"
    command -v aws >/dev/null 2>&1 || fail "aws still not on PATH after install"
    ok "installed: $(aws --version 2>&1)"
fi
# Verify the credentials actually work BEFORE we depend on them for dvc pull.
aws sts get-caller-identity >/dev/null 2>&1 || fail "aws sts get-caller-identity failed — bad/expired credentials or wrong region"
ok "credentials verified: $(aws sts get-caller-identity --query Arn --output text 2>/dev/null)"

# ----------------------------------------------------------------- 4. base tooling
step "[4/9] Base tooling (git, python3, node)"
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
step "[5/9] Claude Code CLI"
if command -v claude >/dev/null 2>&1; then
    ok "already installed: $(claude --version 2>&1 | head -1)"
elif command -v npm >/dev/null 2>&1; then
    $SUDO npm install -g @anthropic-ai/claude-code >/dev/null 2>&1 \
        && ok "installed: $(claude --version 2>&1 | head -1)" \
        || warn "Claude Code install failed (non-fatal — the experiment script runs without it)"
else
    warn "npm unavailable — skipping Claude Code (experiment still runs)"
fi
[ -n "${ANTHROPIC_API_KEY:-}" ] && ok "ANTHROPIC_API_KEY set (claude works non-interactively)" \
    || warn "ANTHROPIC_API_KEY not set — run 'claude' interactively to log in if you want agent debugging"

# ----------------------------------------------------------------- 6. repo
step "[6/9] Repository"
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
step "[7/9] Python stack"
if [ -f optimization/qlora/setup_unsloth_pod.sh ] && python3 -c "import unsloth" 2>/dev/null; then
    echo "  unsloth image detected — running setup_unsloth_pod.sh (validates stack, installs dvc/xgboost)..."
    bash optimization/qlora/setup_unsloth_pod.sh || warn "setup_unsloth_pod.sh reported issues — check the log above"
else
    warn "not on a preconfigured unsloth image — ensuring DVC only (the training stack must come from the GPU image)."
    python3 -c "import dvc" 2>/dev/null || pip3 install --quiet "dvc[s3]" || fail "failed to install dvc[s3]"
fi
python3 -c "import dvc" 2>/dev/null && ok "dvc: $(python3 -c 'import dvc; print(dvc.__version__)' 2>/dev/null)" || fail "dvc not importable"

# ----------------------------------------------------------------- 8. data (dvc pull)
step "[8/9] Data — dvc pull (candles required, dataset for the baseline)"
# Candles are MANDATORY: labeling + trade simulation both read them. The labeled
# dataset.jsonl is the filtered baseline; the unfiltered set is generated locally.
dvc pull backtest/data/candles || fail "dvc pull candles failed — check AWS creds/region and 'dvc remote list'"
ok "candles pulled ($(ls backtest/data/candles/*_1h.json 2>/dev/null | wc -l | tr -d ' ') symbols with 1h data)"
dvc pull backtest/data/labeled/dataset.jsonl 2>/dev/null && ok "filtered baseline dataset pulled" || warn "baseline dataset.jsonl not pulled (only needed for side-by-side comparison)"

# ----------------------------------------------------------------- 9. launch
step "[9/9] Launch experiment"
LAUNCHER="optimization/qlora/run_experiment_no_drawdown_filter.sh"
[ -f "$LAUNCHER" ] || fail "launcher missing: $LAUNCHER (wrong branch?)"
mkdir -p optimization/qlora/logs
RUN_LOG="optimization/qlora/logs/qlora_no_drawdown_filter.log"

if [ "$AUTO_LAUNCH" != "1" ]; then
    step "READY (AUTO_LAUNCH=0) — start it yourself with:"
    echo "    cd $LANGGRAPH && nohup bash $LAUNCHER > $RUN_LOG 2>&1 & echo \"PID: \$!\""
    exit 0
fi

LAUNCH_FLAGS=""; [ "$SMOKE_ONLY" = "1" ] && LAUNCH_FLAGS="--smoke-only"
echo "  launching: bash $LAUNCHER $LAUNCH_FLAGS  (log → $RUN_LOG)"
nohup bash "$LAUNCHER" $LAUNCH_FLAGS > "$RUN_LOG" 2>&1 &
PID=$!
ok "experiment started (PID $PID)"
echo
echo "  Monitor:   tail -f $LANGGRAPH/$RUN_LOG"
echo "  GPU:       watch -n 10 nvidia-smi"
echo "  Debug:     run 'claude' in $LANGGRAPH and point it at"
echo "             .claude/EXPERIMENT_NO_DRAWDOWN_FILTER.md (§7 Agent Runbook)"
echo "  user_data log: $LOG"
