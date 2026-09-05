param(
    [Parameter(Mandatory = $true)]
    [switch]$ConfirmSingleHoldoutUse,
    [string]$DataRoot = ''
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backendRoot = Join-Path $repoRoot 'backend'
$backendPython = Join-Path $backendRoot '.venv\Scripts\python.exe'
$env:TORCH_HOME = Join-Path $env:LOCALAPPDATA 'SmartParking\torch-cache'
$rootArguments = @()
if ($DataRoot) {
    $rootArguments = @('--data-root', $DataRoot, '--model-root', (Join-Path $DataRoot 'models'))
}
Push-Location $backendRoot
try {
    $occupancy = @('-m', 'app.cli.train_occupancy_v3', 'finalize') + $rootArguments
    & $backendPython @occupancy
    if ($LASTEXITCODE -ne 0) { throw 'Protected occupancy holdout failed.' }
    $temporal = @('-m', 'app.cli.tune_video_temporal') + $rootArguments
    & $backendPython @temporal
    if ($LASTEXITCODE -ne 0) { throw 'Validation-only temporal tuning failed.' }
    $localizer = @('-m', 'app.cli.train_slot_localizer', 'finalize') + $rootArguments
    & $backendPython @localizer
    if ($LASTEXITCODE -ne 0) { throw 'Protected localizer holdout failed.' }
    $scenarioAudit = @('-m', 'app.cli.audit_occupancy_scenarios') + $rootArguments
    & $backendPython @scenarioAudit
    if ($LASTEXITCODE -ne 0) { throw 'Prepared-scenario regression audit failed.' }
}
finally { Pop-Location }
