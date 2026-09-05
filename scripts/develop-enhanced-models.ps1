param(
    [ValidateSet('Smoke', 'Standard')]
    [string]$Profile = 'Standard',
    [int]$OccupancyEpochs = 0,
    [int]$LocalizerEpochs = 15,
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
    $occupancy = @(
        '-m', 'app.cli.train_occupancy_v3', 'develop', '--profile', $Profile.ToLowerInvariant()
    ) + $rootArguments
    if ($OccupancyEpochs -gt 0) { $occupancy += @('--epochs', $OccupancyEpochs) }
    & $backendPython @occupancy
    if ($LASTEXITCODE -ne 0) { throw 'Occupancy candidate development failed.' }
    $localizer = @(
        '-m', 'app.cli.train_slot_localizer', 'develop', '--epochs', $LocalizerEpochs
    ) + $rootArguments
    & $backendPython @localizer
    if ($LASTEXITCODE -ne 0) { throw 'Slot-localizer development failed.' }
    $hybrid = @(
        '-m', 'app.cli.train_slot_localizer', 'evaluate-hybrid'
    ) + $rootArguments
    & $backendPython @hybrid
    if ($LASTEXITCODE -ne 0) { throw 'Validation-only layout-localizer selection failed.' }
}
finally { Pop-Location }
