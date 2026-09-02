param(
    [ValidateRange(0.15, 0.40)]
    [double]$ValidationFraction = 0.25,
    [int]$RandomState = 42
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
    & $backendPython -m app.cli.verify_prepared_data
    if ($LASTEXITCODE -ne 0) { throw 'Prepared-data verification failed.' }
    & $backendPython -m app.cli.train_occupancy_model `
        --validation-fraction $ValidationFraction `
        --random-state $RandomState
    if ($LASTEXITCODE -ne 0) { throw "Model training failed with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
}

Write-Host 'Local occupancy model trained successfully.' -ForegroundColor Green
