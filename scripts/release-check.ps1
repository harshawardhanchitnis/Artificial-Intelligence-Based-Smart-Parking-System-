param(
    [string]$ApiBase = 'http://127.0.0.1:8000/api/v1',
    [string]$FrontendBase = 'http://localhost:3000',
    [ValidateRange(1, 120)]
    [int]$TimeoutSeconds = 15,
    [switch]$IncludeInference
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$reliabilityScript = Join-Path $PSScriptRoot 'reliability-check.ps1'

function Invoke-CheckedJson([string]$Uri) {
    $response = Invoke-WebRequest -UseBasicParsing -Uri $Uri -TimeoutSec $TimeoutSeconds
    if ($response.StatusCode -ne 200) {
        throw "GET $Uri returned HTTP $($response.StatusCode)."
    }
    return $response.Content | ConvertFrom-Json
}

Write-Host 'Running complete Version 1.0 runtime reliability check...' -ForegroundColor Cyan
$reliabilityArguments = @(
    '-ExecutionPolicy', 'Bypass',
    '-File', $reliabilityScript,
    '-ApiBase', $ApiBase,
    '-FrontendBase', $FrontendBase,
    '-TimeoutSeconds', $TimeoutSeconds
)
if ($IncludeInference) { $reliabilityArguments += '-IncludeInference' }
& powershell @reliabilityArguments
if ($LASTEXITCODE -ne 0) { throw 'Runtime reliability check failed.' }

Write-Host 'Checking immutable release and presentation contracts...' -ForegroundColor Cyan
$manifest = Invoke-CheckedJson -Uri "$ApiBase/system/release"
$live = Invoke-CheckedJson -Uri "$ApiBase/health/live"
$system = Invoke-CheckedJson -Uri "$ApiBase/system"
$presentation = Invoke-CheckedJson -Uri "$ApiBase/demo/readiness"
$showcase = Invoke-CheckedJson -Uri "$ApiBase/demo/showcase"

$version = (Get-Content -LiteralPath (Join-Path $repoRoot 'VERSION') -Raw).Trim()
if ($version -ne '1.0.0' -or $manifest.version -ne $version) {
    throw 'Version 1.0 identity does not match across the repository and API.'
}
if ($live.version -ne $version -or $system.version -ne $version) {
    throw 'Runtime service version does not match VERSION.'
}
if ($manifest.stage -ne 'final' -or $system.release_stage -ne 'final') {
    throw 'The application is not reporting the final release stage.'
}
if ($manifest.mode -ne 'offline') { throw 'Release mode must remain offline.' }
if (($manifest.datasets -join '|') -ne 'PKLot|CNRPark+EXT|ACPDS') {
    throw 'Approved release dataset order or coverage changed.'
}
if ($manifest.model.name -ne 'parking-occupancy-logistic-v2') {
    throw 'Release model identity changed.'
}
if ([double]$manifest.model.decision_threshold -ne 0.55) {
    throw 'Release decision threshold changed.'
}
if ($manifest.model.unseen_test_samples -ne 1800 -or
    [double]$manifest.model.unseen_test_accuracy -ne 0.906667) {
    throw 'Release unseen-test contract changed.'
}
if ($manifest.boundaries.hardware -or $manifest.boundaries.live_data -or
    $manifest.boundaries.cloud_ai) {
    throw 'Offline product boundaries changed.'
}
if (-not $presentation.ready -or $presentation.showcase_count -ne 3) {
    throw 'Presentation readiness contract failed.'
}
if ($showcase.count -ne 3) { throw 'Showcase no longer contains exactly three scenarios.' }

[pscustomobject]@{
    Release = $manifest.label
    Version = $manifest.version
    Stage = $manifest.stage
    Mode = $manifest.mode
    Datasets = $manifest.datasets -join ', '
    Model = $manifest.model.name
    DecisionThreshold = $manifest.model.decision_threshold
    UnseenTestSamples = $manifest.model.unseen_test_samples
    UnseenTestAccuracy = '{0:P1}' -f [double]$manifest.model.unseen_test_accuracy
    PresentationReady = [bool]$presentation.ready
    ShowcaseScenarios = $showcase.count
    InferenceIncluded = [bool]$IncludeInference
} | Format-List

Write-Host 'Version 1.0 release and presentation handover check passed.' -ForegroundColor Green
