param(
    [Parameter(Mandatory = $true)]
    [string]$ScenarioId
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backendRoot = Join-Path $repoRoot 'backend'
$backendPython = Join-Path $backendRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $backendPython)) {
    throw 'Backend environment is missing. Run scripts\setup.ps1 first.'
}

Push-Location $backendRoot
try {
    & $backendPython -m app.cli.analyze_scenario $ScenarioId
    if ($LASTEXITCODE -ne 0) { throw "Inference failed with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
