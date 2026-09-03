param(
    [string]$ApiBase = 'http://127.0.0.1:8000/api/v1',
    [string]$FrontendBase = 'http://localhost:3000',
    [ValidateRange(1, 120)]
    [int]$TimeoutSeconds = 15,
    [switch]$IncludeInference
)

$ErrorActionPreference = 'Stop'

function Invoke-CheckedJson {
    param(
        [Parameter(Mandatory)] [string]$Uri,
        [ValidateSet('GET', 'POST')] [string]$Method = 'GET'
    )
    $response = Invoke-WebRequest -UseBasicParsing -Method $Method -Uri $Uri -TimeoutSec $TimeoutSeconds
    if ($response.StatusCode -lt 200 -or $response.StatusCode -ge 300) {
        throw "$Method $Uri returned HTTP $($response.StatusCode)."
    }
    return $response.Content | ConvertFrom-Json
}

Write-Host 'Checking backend liveness and readiness...' -ForegroundColor Cyan
$live = Invoke-CheckedJson -Uri "$ApiBase/health/live"
$ready = Invoke-CheckedJson -Uri "$ApiBase/health/ready"
if ($live.status -ne 'alive') { throw 'Backend liveness contract failed.' }
if (-not $ready.ready) { throw 'Backend deep readiness contract failed.' }

Write-Host 'Checking read-only application API contracts...' -ForegroundColor Cyan
$catalogue = Invoke-CheckedJson -Uri "$ApiBase/datasets/catalogue"
$scenarios = Invoke-CheckedJson -Uri "$ApiBase/datasets/scenarios?limit=500"
$model = Invoke-CheckedJson -Uri "$ApiBase/model/status"
$dashboard = Invoke-CheckedJson -Uri "$ApiBase/dashboard/summary"
$history = Invoke-CheckedJson -Uri "$ApiBase/analysis/history?limit=5"
$analytics = Invoke-CheckedJson -Uri "$ApiBase/analytics/summary"
$diagnostics = Invoke-CheckedJson -Uri "$ApiBase/diagnostics/summary"
$showcase = Invoke-CheckedJson -Uri "$ApiBase/demo/showcase"

if (-not $catalogue.prepared -or $catalogue.scenario_count -lt 1) { throw 'Prepared catalogue contract failed.' }
if ($scenarios.count -lt 1) { throw 'Scenario catalogue is empty.' }
if (-not $model.ready) { throw 'Local model contract failed.' }
if (-not $diagnostics.benchmark_available) { throw 'Independent benchmark is unavailable.' }
if ($diagnostics.independent_benchmark.unseen_test.unique_samples -ne 1800) {
    throw 'Independent unseen benchmark sample count changed.'
}
if ($showcase.count -ne 3) { throw 'Showcase must contain exactly three datasets.' }

Write-Host 'Checking every frontend route...' -ForegroundColor Cyan
$frontendRoutes = @(
    '/', '/presentation', '/analyse', '/parking-lots', '/history',
    '/analytics', '/diagnostics', '/reports', '/system'
)
foreach ($route in $frontendRoutes) {
    $response = Invoke-WebRequest -UseBasicParsing -Uri "$FrontendBase$route" -TimeoutSec $TimeoutSeconds
    if ($response.StatusCode -ne 200) { throw "Frontend route $route returned HTTP $($response.StatusCode)." }
}

if ($IncludeInference) {
    Write-Host 'Running one optional end-to-end inference...' -ForegroundColor Cyan
    $scenarioId = [string]$showcase.scenarios[0].id
    $result = Invoke-CheckedJson -Method POST -Uri "$ApiBase/analysis/scenarios/$([Uri]::EscapeDataString($scenarioId))"
    if ($result.total_spaces -ne ($result.occupied_spaces + $result.vacant_spaces)) {
        throw 'Inference occupancy totals are inconsistent.'
    }
    if ($result.analysis_id -lt 1) { throw 'Inference result was not persisted.' }
}

[pscustomobject]@{
    Backend = $live.status
    Readiness = "$($ready.summary.passed)/$($ready.summary.total)"
    Scenarios = $catalogue.scenario_count
    SavedAnalyses = $dashboard.analysis_count
    RecentHistoryRows = $history.count
    AnalyticsRuns = $analytics.total_runs
    UnseenBenchmarkSamples = $diagnostics.independent_benchmark.unseen_test.unique_samples
    FrontendRoutes = $frontendRoutes.Count
    InferenceIncluded = [bool]$IncludeInference
} | Format-List

Write-Host 'Milestone 7 runtime reliability check passed.' -ForegroundColor Green
