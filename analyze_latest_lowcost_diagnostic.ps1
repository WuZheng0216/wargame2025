param(
    [string]$DiagnosticsRoot = ".\test\diagnostics"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonExe = "C:\Users\Tk\.conda\envs\scene\python.exe"

if (-not (Test-Path $pythonExe)) {
    throw "scene 环境 Python 不存在: $pythonExe"
}

Push-Location $repoRoot
try {
    $resolvedDiagnostics = Resolve-Path $DiagnosticsRoot -ErrorAction Stop
    & $pythonExe .\analyze_lowcost_trajectory_diagnostics.py --diagnostics-root $resolvedDiagnostics
    if ($LASTEXITCODE -ne 0) {
        throw "低成本弹轨迹分析失败，exit code=$LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
