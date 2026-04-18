param(
    [Parameter(Mandatory = $true)]
    [int]$Runs,

    [string]$PythonExe = "C:\Users\Tk\.conda\envs\scene\python.exe",

    [string]$BatchRef = "",

    [string]$BatchesDir = "test\batches",

    [string]$RedBatchLearningMode = ""
)

$ErrorActionPreference = "Stop"

function Resolve-BatchRoot {
    param(
        [string]$BatchRefValue,
        [string]$BatchesDirValue
    )

    if ($BatchRefValue -and $BatchRefValue.Trim().Length -gt 0) {
        if (Test-Path -LiteralPath $BatchRefValue) {
            return (Resolve-Path -LiteralPath $BatchRefValue).Path
        }

        $candidate = Join-Path $BatchesDirValue $BatchRefValue
        if (Test-Path -LiteralPath $candidate) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }

        throw "Batch not found: $BatchRefValue"
    }

    if (-not (Test-Path -LiteralPath $BatchesDirValue)) {
        throw "Batches directory not found: $BatchesDirValue"
    }

    $latest = Get-ChildItem -LiteralPath $BatchesDirValue -Directory |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $latest) {
        throw "No batch directories found under $BatchesDirValue"
    }

    return $latest.FullName
}

function Read-JsonFile {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw | ConvertFrom-Json
}

$batchRoot = Resolve-BatchRoot -BatchRefValue $BatchRef -BatchesDirValue $BatchesDir
$batchStatusPath = Join-Path $batchRoot "batch_status.json"
$batchConfigPath = Join-Path $batchRoot "batch_config.json"

$batchStatus = Read-JsonFile -Path $batchStatusPath
$batchConfig = Read-JsonFile -Path $batchConfigPath

if ($batchStatus -and $batchStatus.current_run_id) {
    throw "Batch appears to be running already: $($batchStatus.current_run_id)"
}

if (-not (Test-Path -LiteralPath $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

Write-Host "Resuming batch:" $batchRoot
if ($batchStatus) {
    Write-Host ("Completed: {0}/{1}" -f $batchStatus.runs_completed, $batchStatus.runs_total)
}
if ($batchConfig) {
    Write-Host ("Configured end_simtime={0} engine_host={1} step_interval={2}" -f $batchConfig.end_simtime, $batchConfig.engine_host, $batchConfig.step_interval)
    if ($batchConfig.red_batch_learning_mode) {
        Write-Host ("Configured red_batch_learning_mode={0}" -f $batchConfig.red_batch_learning_mode)
    }
}
Write-Host ("Target total runs: {0}" -f $Runs)
$command = @(
    ".\experiment_runner.py",
    "--resume-batch", $batchRoot,
    "--runs", $Runs
)
if ($RedBatchLearningMode -and $RedBatchLearningMode.Trim().Length -gt 0) {
    $command += @("--red-batch-learning-mode", $RedBatchLearningMode)
}

& $PythonExe @command
