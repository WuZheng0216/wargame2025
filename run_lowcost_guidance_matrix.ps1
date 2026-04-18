param(
    [int]$EndSimtime = 1200,
    [string]$EngineHost = "127.0.0.1",
    [double]$StepInterval = 0.1,
    [string[]]$Presets,
    [double]$DiagnosticIntervalSeconds = 3.0,
    [int]$DesiredLaunches = 18,
    [double]$PostLaunchTailSeconds = 120.0,
    [double]$BlueMoveInterval = 20.0,
    [double]$BlueMoveOffsetLon = 0.05,
    [double]$BlueMoveOffsetLat = 0.035,
    [switch]$OneshotDebug
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = "C:\Users\Tk\.conda\envs\scene\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "scene python not found: $pythonExe"
}

Push-Location $repoRoot
try {
    Write-Host "Running low-cost guidance matrix..."
    Write-Host "  end_simtime=$EndSimtime engine_host=$EngineHost step_interval=$StepInterval desired_launches=$DesiredLaunches diag_interval=$DiagnosticIntervalSeconds blue_move_interval=$BlueMoveInterval"
    if ($Presets -and $Presets.Count -gt 0) {
        Write-Host "  presets=$($Presets -join ', ')"
    }

    $cmd = @(
        ".\run_lowcost_guidance_matrix.py",
        "--end-simtime", $EndSimtime,
        "--engine-host", $EngineHost,
        "--step-interval", $StepInterval,
        "--diagnostic-interval-seconds", $DiagnosticIntervalSeconds,
        "--desired-launches", $DesiredLaunches,
        "--post-launch-tail-seconds", $PostLaunchTailSeconds,
        "--blue-move-interval", $BlueMoveInterval,
        "--blue-move-offset-lon", $BlueMoveOffsetLon,
        "--blue-move-offset-lat", $BlueMoveOffsetLat
    )
    if ($Presets -and $Presets.Count -gt 0) {
        $cmd += "--presets"
        $cmd += $Presets
    }
    if ($OneshotDebug) {
        $cmd += "--oneshot-debug"
    }

    & $pythonExe @cmd
    if ($LASTEXITCODE -ne 0) {
        throw "low-cost guidance matrix failed, exit code=$LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
