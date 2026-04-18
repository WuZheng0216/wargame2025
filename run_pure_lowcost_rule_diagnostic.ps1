param(
    [int]$EndSimtime = 600,
    [string]$EngineHost = "127.0.0.1",
    [double]$StepInterval = 0.1,
    [string]$Strategy = "guide_and_move_then_salvo",
    [int]$SalvoSize = 8,
    [double]$SalvoInterval = 45.0,
    [double]$MoveInterval = 45.0,
    [double]$GuideInterval = 60.0,
    [int]$GuideUnits = 1,
    [double]$GuideHoldSeconds = 0.0,
    [double]$ScoutInterval = 60.0,
    [double]$RefreshThreshold = 35.0,
    [int]$DesiredLaunches = 18,
    [double]$PostLaunchTailSeconds = 120.0,
    [double]$DiagnosticIntervalSeconds = 3.0,
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
    Write-Host "Running pure-rule low-cost diagnostic..."
    Write-Host "  strategy=$Strategy end_simtime=$EndSimtime salvo_size=$SalvoSize salvo_interval=$SalvoInterval guide_units=$GuideUnits guide_hold=$GuideHoldSeconds desired_launches=$DesiredLaunches diag_interval=$DiagnosticIntervalSeconds blue_move_interval=$BlueMoveInterval"

    $cmd = @(
        ".\run_lowcost_rule_diagnostic.py",
        "--end-simtime", $EndSimtime,
        "--engine-host", $EngineHost,
        "--step-interval", $StepInterval,
        "--strategy", $Strategy,
        "--salvo-size", $SalvoSize,
        "--salvo-interval", $SalvoInterval,
        "--move-interval", $MoveInterval,
        "--guide-interval", $GuideInterval,
        "--guide-units", $GuideUnits,
        "--guide-hold-seconds", $GuideHoldSeconds,
        "--scout-interval", $ScoutInterval,
        "--refresh-threshold", $RefreshThreshold,
        "--desired-launches", $DesiredLaunches,
        "--post-launch-tail-seconds", $PostLaunchTailSeconds,
        "--diagnostic-interval-seconds", $DiagnosticIntervalSeconds,
        "--blue-move-interval", $BlueMoveInterval,
        "--blue-move-offset-lon", $BlueMoveOffsetLon,
        "--blue-move-offset-lat", $BlueMoveOffsetLat
    )
    if ($OneshotDebug) {
        $cmd += "--oneshot-debug"
    }

    & $pythonExe @cmd
    if ($LASTEXITCODE -ne 0) {
        throw "pure-rule low-cost diagnostic failed, exit code=$LASTEXITCODE"
    }

    Write-Host ""
    Write-Host "Analyze after run:"
    Write-Host "  powershell -ExecutionPolicy Bypass -File .\analyze_latest_lowcost_diagnostic.ps1"
}
finally {
    Pop-Location
}
