$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backendRoot = Join-Path $repoRoot 'backend'
$backendPython = Join-Path $backendRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $backendPython)) {
    throw 'Backend environment is missing. Run scripts\setup.ps1 first.'
}

Push-Location $backendRoot
try {
    & $backendPython -m app.cli.verify_benchmark
    if ($LASTEXITCODE -ne 0) { throw 'Benchmark data verification failed.' }
    & $backendPython -m app.cli.show_model_benchmark
    if ($LASTEXITCODE -ne 0) { throw 'Independent model benchmark failed.' }
}
finally {
    Pop-Location
}
