param(
    [string]$DataRoot = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backendRoot = Join-Path $repoRoot 'backend'
$frontendRoot = Join-Path $repoRoot 'frontend'
$backendPython = Join-Path $backendRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $backendPython)) {
    throw 'Backend environment is missing. Run scripts\setup.ps1 first.'
}
$rootArguments = @()
if ($DataRoot) {
    $rootArguments = @('--data-root', $DataRoot, '--model-root', (Join-Path $DataRoot 'models'))
}
function Assert-Succeeded([string]$Step) {
    if ($LASTEXITCODE -ne 0) { throw "$Step failed with exit code $LASTEXITCODE." }
}

Push-Location $backendRoot
try {
    & $backendPython -m app.cli.migrate_database
    Assert-Succeeded 'database migration'
    & $backendPython -m ruff check app tests
    Assert-Succeeded 'backend lint'
    & $backendPython -m pytest
    Assert-Succeeded 'backend tests'
    & $backendPython -m app.cli.validate_archives
    Assert-Succeeded 'archive validation'
    & $backendPython -m app.cli.verify_refinement @rootArguments
    Assert-Succeeded 'integrated refinement verification'
}
finally { Pop-Location }

Push-Location $frontendRoot
try {
    npm run lint
    Assert-Succeeded 'frontend lint'
    npm run typecheck
    Assert-Succeeded 'frontend typecheck'
    npm run build
    Assert-Succeeded 'frontend production build'
}
finally { Pop-Location }

Write-Host 'Integrated refinement verification passed.' -ForegroundColor Green
