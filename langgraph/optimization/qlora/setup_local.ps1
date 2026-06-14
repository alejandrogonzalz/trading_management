# setup_local.ps1 — Windows setup for QLoRA fine-tuning (RTX 5070 Ti 16GB)
#
# Creates .venv-finetuning, installs Unsloth + dependencies, verifies CUDA + GPU.
# Equivalent of setup_ec2.sh but for Windows PowerShell.
#
# Usage (from langgraph/ directory):
#   cd C:\Users\alex\projects\trading_management\langgraph
#   powershell -ExecutionPolicy Bypass -File optimization\qlora\setup_local.ps1
#
# After setup:
#   .\.venv-finetuning\Scripts\Activate.ps1
#   dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles
#   .\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py --max-steps 3 --max-eval 10
#
# Requirements:
#   - Windows 10/11
#   - NVIDIA RTX 5070 Ti (or any Blackwell/Ada GPU with 16+ GB)
#   - NVIDIA driver >= R570 (572.xx+ on Windows)
#   - Python 3.10+ on PATH
#   - Git on PATH (for pip install from github)
#   - ~30 GB free disk space

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  QLoRA Local Setup (RTX 5070 Ti / Windows)"
Write-Host "  $(Get-Date)"
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# ---------------------------------------------------------------------------
# 1. Check NVIDIA GPU + driver
# ---------------------------------------------------------------------------
Write-Host "[1/6] Checking GPU + NVIDIA driver..." -ForegroundColor Yellow

$nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
if (-not $nvidiaSmi) {
    Write-Host "  ERROR: nvidia-smi not found. Install NVIDIA drivers (R570+ for RTX 5070 Ti)." -ForegroundColor Red
    Write-Host "  Download: https://www.nvidia.com/download/index.aspx"
    exit 1
}

$gpuInfo = & nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1
Write-Host "  $gpuInfo"

# Check for active processes using the GPU
$gpuProcesses = & nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>&1
if ($gpuProcesses -and $gpuProcesses -notmatch "no running") {
    Write-Host "  WARNING: Other processes are using the GPU:" -ForegroundColor Yellow
    Write-Host "  $gpuProcesses"
    Write-Host "  Close them before training (Ollama, Chrome GPU accel, games, etc.)"
}
Write-Host ""

# ---------------------------------------------------------------------------
# 2. Check Python version (3.10+)
# ---------------------------------------------------------------------------
Write-Host "[2/6] Checking Python..." -ForegroundColor Yellow

$pythonCmd = $null
foreach ($candidate in @("python3.12", "python3.11", "python3.10", "python")) {
    $cmd = Get-Command $candidate -ErrorAction SilentlyContinue
    if ($cmd) {
        $pythonCmd = $cmd.Source
        break
    }
}

if (-not $pythonCmd) {
    Write-Host "  ERROR: Python 3.10+ not found on PATH." -ForegroundColor Red
    Write-Host "  Install from https://www.python.org/downloads/ (check 'Add to PATH')"
    exit 1
}

$pyVersion = & $pythonCmd --version 2>&1
Write-Host "  Found: $pyVersion ($pythonCmd)"

# Verify version >= 3.10
$versionOutput = & $pythonCmd -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>&1
$major, $minor = $versionOutput -split '\.'
if ([int]$major -lt 3 -or ([int]$major -eq 3 -and [int]$minor -lt 10)) {
    Write-Host "  ERROR: Python 3.10+ required, found $versionOutput" -ForegroundColor Red
    exit 1
}
Write-Host ""

# ---------------------------------------------------------------------------
# 3. Check Git (needed for pip install from github)
# ---------------------------------------------------------------------------
Write-Host "[3/6] Checking Git..." -ForegroundColor Yellow

$gitCmd = Get-Command git -ErrorAction SilentlyContinue
if (-not $gitCmd) {
    Write-Host "  ERROR: Git not found on PATH." -ForegroundColor Red
    Write-Host "  Install from https://git-scm.com/download/win"
    exit 1
}

$gitVersion = & git --version 2>&1
Write-Host "  Found: $gitVersion"
Write-Host ""

# ---------------------------------------------------------------------------
# 4. Create virtual environment
# ---------------------------------------------------------------------------
Write-Host "[4/6] Creating virtual environment (.venv-finetuning)..." -ForegroundColor Yellow

$venvPath = ".venv-finetuning"

if (Test-Path $venvPath) {
    Write-Host "  Existing venv found. Removing and recreating..."
    Remove-Item -Recurse -Force $venvPath
}

& $pythonCmd -m venv $venvPath
if (-not $?) {
    Write-Host "  ERROR: Failed to create venv." -ForegroundColor Red
    exit 1
}

$pipExe = "$venvPath\Scripts\pip.exe"
$pythonExe = "$venvPath\Scripts\python.exe"

Write-Host "  Created: $venvPath"
Write-Host "  Python: $pythonExe"

# Upgrade pip
Write-Host "  Upgrading pip..."
& $pythonExe -m pip install --upgrade pip --quiet
Write-Host ""

# ---------------------------------------------------------------------------
# 5. Install dependencies
# ---------------------------------------------------------------------------
Write-Host "[5/6] Installing training dependencies..." -ForegroundColor Yellow
Write-Host "  This may take 15-20 minutes (downloads ~8 GB: PyTorch is fetched twice —"
Write-Host "  Unsloth's CPU wheel, then the cu128 CUDA build that replaces it)."
Write-Host ""

# Unsloth from git — resolves its own torch, TRL, transformers, peft, bitsandbytes
Write-Host "  Installing Unsloth (includes PyTorch, TRL, transformers, peft, bitsandbytes)..."
& $pipExe install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
if (-not $?) {
    Write-Host "  ERROR: Unsloth installation failed." -ForegroundColor Red
    Write-Host "  Check internet connection and that git is on PATH."
    exit 1
}

# CRITICAL (Windows): Unsloth resolves a CPU-only torch from PyPI. Unlike the EC2
# Deep Learning AMI (which ships a matched CUDA driver so the CUDA wheel resolves
# automatically), Windows has no CUDA in the wheel index, so we MUST reinstall the
# CUDA 12.8 build explicitly. The RTX 5070 Ti is Blackwell (sm_120) and ONLY runs
# on cu128 wheels — older CUDA builds won't even load the GPU.
Write-Host "  Reinstalling PyTorch with CUDA 12.8 (cu128) for the RTX 5070 Ti..."
& $pipExe install torch torchvision torchaudio `
    --index-url https://download.pytorch.org/whl/cu128 --upgrade --force-reinstall
if (-not $?) {
    Write-Host "  ERROR: CUDA torch (cu128) installation failed." -ForegroundColor Red
    Write-Host "  Retry manually: $pipExe install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --upgrade --force-reinstall"
    exit 1
}

# Force-reinstalling torch can leave bitsandbytes linked against the old build.
# Reinstall it against the fresh CUDA torch so 4-bit quantization works.
Write-Host "  Reinstalling bitsandbytes against the CUDA torch build..."
& $pipExe install bitsandbytes --upgrade --force-reinstall --quiet
if (-not $?) {
    Write-Host "  WARNING: bitsandbytes reinstall failed — 4-bit loading may break." -ForegroundColor Yellow
}

# torch's --force-reinstall pulls the LATEST fsspec, which breaks datasets
# (needs <=2025.9.0) and s3fs/DVC (needs ==2025.9.0). Pin it back so dataset
# loading and `dvc pull` work. 2025.9.0 satisfies torch, datasets, and s3fs.
Write-Host "  Pinning fsspec==2025.9.0 (datasets + s3fs/DVC compatibility)..."
& $pipExe install "fsspec[http]==2025.9.0" --quiet
if (-not $?) {
    Write-Host "  WARNING: fsspec pin failed — 'dvc pull' or dataset loading may break." -ForegroundColor Yellow
}

# Additional deps not included by Unsloth
Write-Host "  Installing DVC, scikit-learn, and utilities..."
& $pipExe install "dvc[s3]" scikit-learn pyyaml tqdm matplotlib httpx --quiet
if (-not $?) {
    Write-Host "  WARNING: Some optional dependencies failed to install." -ForegroundColor Yellow
}

Write-Host "  Done."
Write-Host ""

# ---------------------------------------------------------------------------
# 6. Verify installation
# ---------------------------------------------------------------------------
Write-Host "[6/6] Verifying installation..." -ForegroundColor Yellow

$verifyScript = @"
import sys
print(f'  Python: {sys.version}')

import torch
cuda_ok = torch.cuda.is_available()
print(f'  CUDA available: {cuda_ok}')
if cuda_ok:
    print(f'  GPU: {torch.cuda.get_device_name(0)}')
    print(f'  Compute capability: {torch.cuda.get_device_capability(0)}')
    print(f'  PyTorch: {torch.__version__}')
    print(f'  CUDA version: {torch.version.cuda}')
    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'  VRAM: {vram_gb:.1f} GB')
else:
    print('  ERROR: CUDA not available. Training will not work.')
    print('  Fix: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 --upgrade --force-reinstall')
    sys.exit(1)

import bitsandbytes
print(f'  bitsandbytes: {bitsandbytes.__version__}')

try:
    from unsloth import FastLanguageModel
    print('  Unsloth: OK')
except Exception as e:
    print(f'  Unsloth: FAILED ({e})')
    sys.exit(1)

import trl, transformers, peft
print(f'  TRL: {trl.__version__}  transformers: {transformers.__version__}  peft: {peft.__version__}')

try:
    import dvc
    print(f'  DVC: OK')
except ImportError:
    print('  DVC: not installed (pull data manually with aws s3 cp)')

print()
print('  All checks passed.')
"@

& $pythonExe -c $verifyScript
if (-not $?) {
    Write-Host ""
    Write-Host "  Verification failed. See errors above." -ForegroundColor Red
    exit 1
}

# Create log and result directories
New-Item -ItemType Directory -Force -Path "optimization\qlora\logs" | Out-Null
New-Item -ItemType Directory -Force -Path "optimization\qlora\results" | Out-Null

Write-Host ""
Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Setup complete. Next steps:" -ForegroundColor Green
Write-Host ""
Write-Host "  1. Activate the venv:"
Write-Host "     .\.venv-finetuning\Scripts\Activate.ps1"
Write-Host ""
Write-Host "  2. Pull dataset + candles (REQUIRED for trade metrics):"
Write-Host "     dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles"
Write-Host ""
Write-Host "  3. Smoke test (5-10 min):"
Write-Host "     .\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py ``"
Write-Host "       --max-steps 3 --max-eval 10 --diagnostic-samples 0"
Write-Host ""
Write-Host "  4. Run training (see LOCAL_GUIDE.md for full command):"
Write-Host "     .\.venv-finetuning\Scripts\python.exe optimization\qlora\train_qlora.py ``"
Write-Host "       --lr 0.00005 --rank 8 --alpha 16 --epochs 1 --max-steps 1500 ``"
Write-Host "       --batch-size 1 --grad-accum 16 --max-eval 200 --diagnostic-samples 0 ``"
Write-Host "       --tag qlora_local --output-dir backtest\data\models\qlora_local ``"
Write-Host "       2>&1 | Tee-Object -FilePath optimization\qlora\logs\qlora_local.log"
Write-Host ""
Write-Host "  IMPORTANT: Disable sleep before overnight training!"
Write-Host "  Settings > System > Power > Sleep = Never"
Write-Host "============================================================" -ForegroundColor Green
