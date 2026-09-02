$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backendRoot = Join-Path $repoRoot 'backend'
$frontendRoot = Join-Path $repoRoot 'frontend'
$virtualEnvironment = Join-Path $backendRoot '.venv'
$backendPython = Join-Path $virtualEnvironment 'Scripts\python.exe'

function Assert-CommandSucceeded([string]$step) {
    if ($LASTEXITCODE -ne 0) {
        throw "$step failed with exit code $LASTEXITCODE."
    }
}

Write-Host 'Preparing Smart Parking foundation...' -ForegroundColor Cyan

if (-not (Test-Path -LiteralPath (Join-Path $repoRoot '.env'))) {
    Copy-Item -LiteralPath (Join-Path $repoRoot '.env.example') -Destination (Join-Path $repoRoot '.env')
    Write-Host 'Created .env from .env.example'
}

if (-not (Test-Path -LiteralPath $backendPython)) {
    py -3.13 -m venv $virtualEnvironment
    Assert-CommandSucceeded 'Python virtual-environment creation'
}

Push-Location $backendRoot
try {
    & $backendPython -m pip install --upgrade pip
    Assert-CommandSucceeded 'pip upgrade'
    & $backendPython -m pip install -e '.[dev]'
    Assert-CommandSucceeded 'backend dependency installation'
    & $backendPython -m app.cli.migrate_database
    Assert-CommandSucceeded 'database migration'
    & $backendPython -m pytest
    Assert-CommandSucceeded 'backend tests'
    & $backendPython -m app.cli.validate_archives
    Assert-CommandSucceeded 'dataset archive validation'
}
finally {
    Pop-Location
}

Push-Location $frontendRoot
try {
    if (Test-Path -LiteralPath (Join-Path $frontendRoot 'package-lock.json')) {
        npm ci
        Assert-CommandSucceeded 'frontend dependency installation'
    }
    else {
        npm install
        Assert-CommandSucceeded 'frontend dependency installation'
    }
    npm run lint
    Assert-CommandSucceeded 'frontend lint'
    npm run typecheck
    Assert-CommandSucceeded 'frontend type check'
}
finally {
    Pop-Location
}

Write-Host ''
Write-Host 'Milestone 4 setup completed successfully.' -ForegroundColor Green
Write-Host 'Use scripts\start-backend.ps1 and scripts\start-frontend.ps1 in two terminals.'
