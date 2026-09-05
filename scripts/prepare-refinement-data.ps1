param(
    [ValidateSet('Smoke', 'Standard')]
    [string]$Profile = 'Standard',
    [string]$ArtifactRoot = '',
    [switch]$PrepareVideos
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backendRoot = Join-Path $repoRoot 'backend'
$backendPython = Join-Path $backendRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $backendPython)) {
    throw 'Backend environment is missing. Run scripts\setup.ps1 first.'
}
$arguments = @('-m', 'app.cli.prepare_v2_protocol', '--profile', $Profile.ToLowerInvariant())
if ($ArtifactRoot) { $arguments += @('--artifact-root', $ArtifactRoot) }
Push-Location $backendRoot
try {
    $scenarioArguments = @(
        '-m', 'app.cli.prepare_datasets', 'demo', '--samples-per-dataset', '10'
    )
    if ($ArtifactRoot) { $scenarioArguments += @('--output-root', $ArtifactRoot) }
    & $backendPython @scenarioArguments
    if ($LASTEXITCODE -ne 0) { throw 'Thirty-scenario catalogue preparation failed.' }
    $scenarioRootArguments = @()
    if ($ArtifactRoot) { $scenarioRootArguments = @('--data-root', $ArtifactRoot) }
    & $backendPython -m app.cli.audit_scenarios @scenarioRootArguments
    if ($LASTEXITCODE -ne 0) { throw 'Scenario overlay QC generation failed.' }
    Write-Warning 'Review all three generated contact sheets, then rerun audit_scenarios with --confirm-reviewer before final verification.'
    & $backendPython @arguments
    if ($LASTEXITCODE -ne 0) { throw 'Leakage-safe corpus preparation failed.' }
    if ($PrepareVideos) {
        if ($ArtifactRoot) {
            & $backendPython -m app.cli.prepare_demo_videos --data-root $ArtifactRoot
        }
        else {
            & $backendPython -m app.cli.prepare_demo_videos
        }
        if ($LASTEXITCODE -ne 0) { throw 'Prepared-video generation failed.' }
    }
}
finally { Pop-Location }
