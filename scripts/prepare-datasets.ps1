param(
    [ValidateSet('Plan', 'Demo', 'Full')]
    [string]$Profile = 'Demo',
    [ValidateRange(1, 30)]
    [int]$SamplesPerDataset = 3,
    [switch]$Force,
    [switch]$ConfirmFullExtraction
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$backendRoot = Join-Path $repoRoot 'backend'
$backendPython = Join-Path $backendRoot '.venv\Scripts\python.exe'

if (-not (Test-Path -LiteralPath $backendPython)) {
    throw 'Backend environment is missing. Run scripts\setup.ps1 first.'
}

$arguments = @('-m', 'app.cli.prepare_datasets')
switch ($Profile) {
    'Plan' { $arguments += 'plan' }
    'Demo' {
        $arguments += @('demo', '--samples-per-dataset', $SamplesPerDataset)
        if ($Force) { $arguments += '--force' }
    }
    'Full' {
        $arguments += 'full'
        if ($ConfirmFullExtraction) { $arguments += '--confirm-full-extraction' }
        if ($Force) { $arguments += '--force' }
    }
}

Push-Location $backendRoot
try {
    & $backendPython @arguments
    if ($LASTEXITCODE -ne 0) { throw "Dataset preparation failed with exit code $LASTEXITCODE." }
}
finally {
    Pop-Location
}
