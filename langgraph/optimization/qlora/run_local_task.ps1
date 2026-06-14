# run_local_task.ps1 — run the QLoRA local training as a Windows SCHEDULED TASK.
#
# WHY: over SSH (and on logoff), Windows terminates the whole session's process
# tree when you disconnect — that is what killed the 4am run, NOT sleep (sleep is
# already disabled: powercfg shows "Sleep Idle State Disabled"). A scheduled task
# set to "run whether the user is logged on or not" runs detached from your SSH
# session, so it survives disconnect AND logoff.
#
# The task runs in a NON-INTERACTIVE session (session 0). On consumer GPUs CUDA
# usually works there, but not always — so run -GpuCheck FIRST to confirm
# torch.cuda.is_available() == True in that context before the long run.
#
# REQUIRED before training (or financial metrics come out 0):
#   dvc pull backtest/data/labeled/dataset.jsonl backtest/data/candles
#
# Usage — run these in YOUR interactive SSH PowerShell (you'll be prompted for
# your Windows password; it is stored in the task, never shown to anyone):
#
#   .\optimization\qlora\run_local_task.ps1 -GpuCheck   # 1) verify GPU in task session
#   .\optimization\qlora\run_local_task.ps1             # 2) register + start training
#   .\optimization\qlora\run_local_task.ps1 -Status     # check state + tail the log
#   .\optimization\qlora\run_local_task.ps1 -Stop       # stop + remove the task
#   .\optimization\qlora\run_local_task.ps1 --resume    # resume from last checkpoint
#
# After step 2 you can safely disconnect SSH. Reconnect anytime and use -Status,
# or:  Get-Content optimization\qlora\logs\qlora_local.log -Wait -Tail 40

[CmdletBinding()]
param(
    [switch]$GpuCheck,
    [switch]$Stop,
    [switch]$Status,
    [string]$TaskName = "QLoRA_Local_Train",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

$ScriptDir     = $PSScriptRoot
$LangGraphRoot = (Resolve-Path (Join-Path $ScriptDir "..\..")).Path
$RunLocal      = Join-Path $ScriptDir "run_local.ps1"
$Python        = Join-Path $LangGraphRoot ".venv-finetuning\Scripts\python.exe"
$LogDir        = Join-Path $ScriptDir "logs"
$LogFile       = Join-Path $LogDir "qlora_local.log"
$GpuLog        = Join-Path $LogDir "gpu_check.log"
$GpuTask       = "QLoRA_GpuCheck"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null

# ---- -Status: report task state + tail the training log, then exit ----
if ($Status) {
    $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if (-not $t) { Write-Host "Task '$TaskName' is not registered." -ForegroundColor Yellow }
    else {
        $info = Get-ScheduledTaskInfo -TaskName $TaskName
        Write-Host "Task '$TaskName': State=$($t.State)  LastRun=$($info.LastRunTime)  LastResult=0x$('{0:X}' -f $info.LastTaskResult)"
    }
    if (Test-Path $LogFile) {
        Write-Host "--- tail $LogFile ---" -ForegroundColor Cyan
        Get-Content $LogFile -Tail 25
    } else { Write-Host "No log yet at $LogFile" -ForegroundColor Yellow }
    return
}

# ---- -Stop: stop + remove the task, then exit ----
if ($Stop) {
    foreach ($name in @($TaskName, $GpuTask)) {
        if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
            Stop-ScheduledTask    -TaskName $name -ErrorAction SilentlyContinue
            Unregister-ScheduledTask -TaskName $name -Confirm:$false
            Write-Host "Removed task '$name'." -ForegroundColor Green
        }
    }
    return
}

if (-not (Test-Path $Python)) {
    Write-Host "ERROR: venv python not found at $Python" -ForegroundColor Red
    Write-Host "  Run optimization\qlora\setup_local.ps1 first."
    exit 1
}

# ---- credential (needed for "run whether logged on or not") ----
$defaultUser = "$env:USERDOMAIN\$env:USERNAME"
Write-Host "Enter the Windows password for $defaultUser (stored in the task; not displayed)." -ForegroundColor Cyan
$cred = Get-Credential -UserName $defaultUser -Message "Windows account password for the scheduled task"
$user = $cred.UserName
$pass = $cred.GetNetworkCredential().Password

# Settings shared by both tasks: no time limit, survive everything, don't double-run.
$settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -StartWhenAvailable -DontStopOnIdleEnd `
    -ExecutionTimeLimit ([TimeSpan]::Zero) `
    -MultipleInstances IgnoreNew

function Register-And-Run([string]$name, [Microsoft.Management.Infrastructure.CimInstance]$action) {
    if (Get-ScheduledTask -TaskName $name -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $name -Confirm:$false
    }
    Register-ScheduledTask -TaskName $name -Action $action -Settings $settings `
        -User $user -Password $pass -RunLevel Highest | Out-Null
    Start-ScheduledTask -TaskName $name
}

# ---- -GpuCheck: confirm CUDA is visible in the task's session-0 context ----
if ($GpuCheck) {
    Remove-Item $GpuLog -ErrorAction SilentlyContinue
    $pyArg = "-c `"import torch; open(r'$GpuLog','w').write('cuda=' + str(torch.cuda.is_available()) + ' dev=' + (torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE'))`""
    $action = New-ScheduledTaskAction -Execute $Python -Argument $pyArg -WorkingDirectory $LangGraphRoot
    Write-Host "Running GPU visibility check as a scheduled task..." -ForegroundColor Cyan
    Register-And-Run $GpuTask $action

    for ($i = 0; $i -lt 60 -and -not (Test-Path $GpuLog); $i++) { Start-Sleep -Seconds 1 }
    Start-Sleep -Seconds 1
    Unregister-ScheduledTask -TaskName $GpuTask -Confirm:$false -ErrorAction SilentlyContinue
    if (Test-Path $GpuLog) {
        $res = Get-Content $GpuLog -Raw
        Write-Host "RESULT: $res"
        if ($res -match "cuda=True") {
            Write-Host "GPU is visible in the task session. Safe to launch the real run." -ForegroundColor Green
        } else {
            Write-Host "GPU NOT visible in session 0 — the scheduled-task path won't work for training." -ForegroundColor Red
            Write-Host "  Fall back to the cloud run (run_cloud.sh) or keep an SSH session alive." -ForegroundColor Yellow
        }
    } else {
        Write-Host "GPU check produced no output (task may have failed to start)." -ForegroundColor Red
    }
    return
}

# ---- default: register + start the real training run, detached ----
$runArg = "-NoProfile -ExecutionPolicy Bypass -File `"$RunLocal`""
if ($ExtraArgs) { $runArg += " " + ($ExtraArgs -join " ") }
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $runArg -WorkingDirectory $LangGraphRoot

Write-Host "Registering + starting '$TaskName' (survives SSH disconnect / logoff)..." -ForegroundColor Cyan
Register-And-Run $TaskName $action
Start-Sleep -Seconds 3

$info = Get-ScheduledTaskInfo -TaskName $TaskName
Write-Host "Started. State=$((Get-ScheduledTask -TaskName $TaskName).State)  LastRun=$($info.LastRunTime)" -ForegroundColor Green
Write-Host ""
Write-Host "You can now disconnect SSH safely. To monitor on reconnect:" -ForegroundColor Cyan
Write-Host "  .\optimization\qlora\run_local_task.ps1 -Status"
Write-Host "  Get-Content `"$LogFile`" -Wait -Tail 40"
Write-Host "To stop:  .\optimization\qlora\run_local_task.ps1 -Stop"
