$ErrorActionPreference = 'Stop'

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$frontendRoot = Join-Path $repoRoot 'frontend'

if (-not (Test-Path -LiteralPath (Join-Path $frontendRoot 'node_modules'))) {
    throw 'Frontend dependencies are missing. Run scripts\setup.ps1 first.'
}

Set-Location -LiteralPath $frontendRoot
npm run dev
