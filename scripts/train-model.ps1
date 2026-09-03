param([int]$RandomState = 42)

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
    & $backendPython -m app.cli.verify_benchmark
    if ($LASTEXITCODE -ne 0) { throw 'ML benchmark verification failed.' }
    & $backendPython -m app.cli.train_occupancy_model `
        --random-state $RandomState
    if ($LASTEXITCODE -ne 0) { throw "Model training failed with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
}

Write-Host 'Local occupancy model trained successfully.' -ForegroundColor Green
