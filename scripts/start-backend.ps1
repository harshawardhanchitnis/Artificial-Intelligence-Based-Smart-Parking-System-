$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backendRoot = Join-Path $repoRoot 'backend'
$backendPython = Join-Path $backendRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $backendPython)) {
    throw 'Backend environment is missing. Run scripts\setup.ps1 first.'
}

Set-Location -LiteralPath $backendRoot
& $backendPython -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
