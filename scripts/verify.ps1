$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backendRoot = Join-Path $repoRoot 'backend'
$frontendRoot = Join-Path $repoRoot 'frontend'
$backendPython = Join-Path $backendRoot '.venv\Scripts\python.exe'

function Assert-CommandSucceeded([string]$step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$step failed with exit code $LASTEXITCODE."
    }
}

if (-not (Test-Path -LiteralPath $backendPython)) {
    throw 'Backend environment is missing. Run scripts\setup.ps1 first.'
}

Write-Host 'Checking backend...' -ForegroundColor Cyan
Push-Location $backendRoot
try {
    & $backendPython -m app.cli.migrate_database
    Assert-CommandSucceeded 'database migration'
    & $backendPython -m ruff check app tests
    Assert-CommandSucceeded 'backend lint'
    & $backendPython -m pytest
    Assert-CommandSucceeded 'backend tests'
    & $backendPython -m app.cli.validate_archives
    Assert-CommandSucceeded 'dataset archive validation'
    & $backendPython -m app.cli.verify_prepared_data
    Assert-CommandSucceeded 'prepared demo verification'
    & $backendPython -m app.cli.verify_benchmark
    Assert-CommandSucceeded 'leakage-safe benchmark verification'
    & $backendPython -m app.cli.verify_occupancy_model
    Assert-CommandSucceeded 'local occupancy model verification'
    & $backendPython -m app.cli.show_model_benchmark
    Assert-CommandSucceeded 'independent model benchmark'
    & $backendPython -m app.cli.verify_demo
    Assert-CommandSucceeded 'offline presentation preflight'
    & $backendPython -m app.cli.verify_reliability
    Assert-CommandSucceeded 'deep application readiness'
    & $backendPython -m app.cli.verify_release
    Assert-CommandSucceeded 'Version 1.0 release contract'
}
finally {
    Pop-Location
}

Write-Host 'Checking frontend...' -ForegroundColor Cyan
Push-Location $frontendRoot
try {
    npm run lint
    Assert-CommandSucceeded 'frontend lint'
    npm run typecheck
    Assert-CommandSucceeded 'frontend type check'
    npm run build
    Assert-CommandSucceeded 'frontend production build'
}
finally {
    Pop-Location
}

Write-Host 'All Version 1.0 release checks passed.' -ForegroundColor Green
