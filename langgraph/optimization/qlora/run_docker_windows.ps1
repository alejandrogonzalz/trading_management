# QLoRA Local Training — PowerShell launcher for Windows
#
# Prerequisites:
#   - Docker Desktop with WSL2 backend
#   - NVIDIA GPU driver (latest) installed on Windows
#   - NVIDIA Container Toolkit installed in WSL2 (see docker-compose.local.yml header)
#   - Verify: docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
#
# Usage (from the langgraph/ directory):
#   .\optimization\qlora\run_docker_windows.ps1
#
# With custom flags:
#   .\optimization\qlora\run_docker_windows.ps1 --tag config2 --lr 0.00005 --rank 8 --epochs 3
#
# Interactive shell (for debugging):
#   .\optimization\qlora\run_docker_windows.ps1 -Shell

param(
    [switch]$Shell,
    [switch]$Pull,
    [Parameter(ValueFromRemainingArguments)]
    [string[]]$ExtraFlags
)

$IMAGE = "docker.io/unsloth/unsloth:latest"
$WORKSPACE = (Get-Location).Path

# Pull the image first if requested or if not present
if ($Pull) {
    Write-Host "Pulling $IMAGE (~25 GB, may take a while)..." -ForegroundColor Cyan
    docker pull $IMAGE
}

$dockerArgs = @(
    "run", "--rm",
    "--gpus", "all",
    "--ipc=host",
    "--ulimit", "memlock=-1",
    "--ulimit", "stack=67108864",
    "-v", "${WORKSPACE}:/workspace",
    "-w", "/workspace",
    "-e", "NVIDIA_VISIBLE_DEVICES=all",
    "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility",
    "-e", "OMP_NUM_THREADS=1"
)

if ($Shell) {
    Write-Host "Opening interactive shell in $IMAGE..." -ForegroundColor Cyan
    Write-Host "  nvidia-smi to check GPU"
    Write-Host "  python3 -c 'import flash_attn; print(flash_attn.__version__)' to check FA2"
    Write-Host ""
    $dockerArgs += @("-it", $IMAGE, "bash")
    & docker @dockerArgs
} else {
    $flags = ($ExtraFlags -join " ")
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host "  QLoRA Docker Local — $IMAGE"
    Write-Host "  Workspace: $WORKSPACE"
    Write-Host "  Extra flags: $flags"
    Write-Host "============================================================" -ForegroundColor Green
    Write-Host ""

    $cmd = @"
echo '=== GPU Check ===' &&
nvidia-smi &&
echo '' &&
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')" &&
python3 -c "import flash_attn; print(f'FlashAttention-2: {flash_attn.__version__}')" 2>/dev/null || echo 'FA2: not available' &&
echo '' &&
pip install -q dvc scikit-learn xgboost httpx tqdm pyyaml matplotlib 2>/dev/null &&
echo '' &&
bash optimization/qlora/run_docker_local.sh $flags
"@

    $dockerArgs += @($IMAGE, "bash", "-c", $cmd)
    & docker @dockerArgs
}
