# run_local.ps1 — QLoRA single LOCAL training run (RTX 5070 Ti 16GB), Windows PowerShell.
#
# Windows-native equivalent of run_local.sh. Same fixed strict-temporal split,
# 1 epoch capped at 1500 steps, batch=1 / grad_accum=16 (forced by 16GB VRAM).
# Streams to the console AND tees to optimization\qlora\logs\qlora_local.log.
#
# REQUIRED before running (or financial metrics come out 0):
#   dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles
#
# Usage (from anywhere — the script cd's to the langgraph root itself):
#   .\optimization\qlora\run_local.ps1
#   .\optimization\qlora\run_local.ps1 --resume      # forwards extra flags to train_qlora.py
#
# Detached (the "tmux" equivalent — survives closing the terminal, NOT a logoff):
#   Start-Process powershell -WindowStyle Hidden -ArgumentList `
#     '-NoProfile','-ExecutionPolicy','Bypass','-File', `
#     'C:\Users\alex\projects\trading_management\langgraph\optimization\qlora\run_local.ps1'
#   # then follow it:  Get-Content ...\logs\qlora_local.log -Wait -Tail 40
#
# Flags explained:
#   --lr 0.00005        Higher LR (5e-5) — compensates for only 1 epoch (fewer total updates).
#   --rank 8            Smaller LoRA (~20M params vs 40M in cloud). Fits 16GB comfortably.
#   --alpha 16          2xrank (standard).
#   --epochs 1          One pass, capped at 1500 steps (~60% of an epoch).
#   --batch-size 1      Only value that fits 1024-token samples in 16GB.
#   --grad-accum 16     Effective batch = 1x16 = 16 (same as cloud).
#   --max-steps 1500    Cap at 1500 steps. Learns the main patterns without the full
#                       ~19h commitment. Re-run with --resume to continue to ~2451.
#   --max-eval 200      Evaluate 200 test samples (~18/symbol). Light but representative.
#   --diagnostic-samples 0  Skip train/val gap probes (saves ~2h). Run cloud for full diag.
#   --tag qlora_local   Names the output files.
#   $ExtraArgs          Forwards any extra flags you pass (e.g. --resume).
#
# Expected timing (RTX 5070 Ti, no FA2, Triton kernels):
#   Training:  ~11.7h (1500 steps x 28s/step)
#   Eval:      ~2h (200 generates x ~35s)
#   GGUF:      ~15 min
#   TOTAL:     ~14h (overnight)   COST: $0 (local hardware)
#
# IMPORTANT: disable sleep first — Settings > System > Power > Sleep = Never.

param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir     = $PSScriptRoot
$LangGraphRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$LogDir        = Join-Path $ScriptDir "logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
Set-Location $LangGraphRoot

$Tag     = "qlora_local"
$Python  = Join-Path $LangGraphRoot ".venv-finetuning\Scripts\python.exe"
$Train   = Join-Path $ScriptDir "train_qlora.py"
$LogFile = Join-Path $LogDir "$Tag.log"

if (-not (Test-Path $Python)) {
    Write-Host "ERROR: venv python not found at $Python" -ForegroundColor Red
    Write-Host "  Run optimization\qlora\setup_local.ps1 first."
    exit 1
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  QLoRA - single local run ($Tag)  [RTX 5070 Ti 16GB]"
Write-Host "  Started:     $(Get-Date)"
Write-Host "  Working dir: $PWD"
Write-Host "  Log:         $LogFile"
if ($ExtraArgs) { Write-Host "  Extra args:  $($ExtraArgs -join ' ')" }
Write-Host "============================================================" -ForegroundColor Cyan

& $Python $Train `
    --lr 0.00005 `
    --rank 8 `
    --alpha 16 `
    --epochs 1 `
    --max-steps 1500 `
    --batch-size 1 `
    --grad-accum 16 `
    --max-eval 200 `
    --diagnostic-samples 0 `
    --tag $Tag `
    --output-dir "backtest\data\models\$Tag" `
    @ExtraArgs 2>&1 | Tee-Object -FilePath $LogFile

$code = $LASTEXITCODE

Write-Host ""
if ($code -eq 0) {
    Write-Host "  DONE - $(Get-Date)" -ForegroundColor Green
    Write-Host "  Result:     optimization\qlora\results\qlora_$Tag.json (+ canonical qlora_optimization.json)"
    Write-Host "  Loss curve: optimization\qlora\results\${Tag}_loss_curve.{json,png}"
    Write-Host "  Model:      backtest\data\models\$Tag\ (adapters + GGUF)"
} else {
    Write-Host "  FAILED (exit $code) - see $LogFile" -ForegroundColor Red
}
exit $code
