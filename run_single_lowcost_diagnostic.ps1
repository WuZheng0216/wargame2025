param(
    [int]$EndSimtime = 600,
    [string]$EngineHost = "127.0.0.1",
    [double]$StepInterval = 0.1,
    [double]$TrajectoryInterval = 1.0,
    [switch]$OneshotDebug
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = "C:\Users\Tk\.conda\envs\scene\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "scene 环境 Python 不存在: $pythonExe"
}

Push-Location $repoRoot
try {
    $env:TRAJECTORY_DIAGNOSTICS_ENABLED = "1"
    $env:TRAJECTORY_DIAGNOSTICS_INTERVAL_SECONDS = [string]$TrajectoryInterval
    if ($OneshotDebug) {
        $env:JSQLSIM_ONESHOT_DEBUG = "1"
    } else {
        Remove-Item Env:JSQLSIM_ONESHOT_DEBUG -ErrorAction SilentlyContinue
    }

    Write-Host "运行单局低成本弹诊断..."
    Write-Host "  end_simtime=$EndSimtime engine_host=$EngineHost step_interval=$StepInterval trajectory_interval=$TrajectoryInterval oneshot_debug=$($OneshotDebug.IsPresent)"

    & $pythonExe .\main.py --end-simtime $EndSimtime --engine-host $EngineHost --step-interval $StepInterval
    if ($LASTEXITCODE -ne 0) {
        throw "单局诊断运行失败，exit code=$LASTEXITCODE"
    }

    $diagnosticsDir = Join-Path $repoRoot "test\diagnostics"
    if (Test-Path $diagnosticsDir) {
        Write-Host ""
        Write-Host "最新诊断文件:"
        Get-ChildItem $diagnosticsDir -Filter "trajectory_*.jsonl" |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 4 FullName, LastWriteTime, Length |
            Format-Table -AutoSize
    }

    Write-Host ""
    Write-Host "下一步可直接分析:"
    Write-Host "  .\analyze_latest_lowcost_diagnostic.ps1"
}
finally {
    Pop-Location
}
