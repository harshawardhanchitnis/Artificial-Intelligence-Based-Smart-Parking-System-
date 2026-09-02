# Milestone 4 — Operational Analytics and Export Reports

Milestone 4 turns saved local-AI runs into presentation-ready operational insight while preserving the Milestone 2 catalogue and Milestone 3 inference pipeline.

## Included

- Additive SQLite schema migration; existing analysis rows are preserved.
- Model name, confidence, agreement, and compact per-slot predictions stored for new runs.
- Live dashboard summary backed by the prepared catalogue, model status, and SQLite.
- Occupancy analytics by dataset and across the ten most recent analyses.
- Full-history CSV export and per-analysis CSV/JSON export.
- Enhanced history view with AI quality metrics and clear legacy-row handling.
- Migration and analytics tests plus updated verification scripts.

## Database migration

The migration only adds nullable columns. It can be run repeatedly and does not delete or replace rows:

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\migrate-database.ps1'
```

Backend startup applies the same migration automatically.

## API additions

- `GET /api/v1/dashboard/summary`
- `GET /api/v1/analytics/summary`
- `GET /api/v1/reports/analyses.csv`
- `GET /api/v1/reports/analyses/{analysis_id}.csv`
- `GET /api/v1/reports/analyses/{analysis_id}.json`

Exports are generated on demand and do not create report files inside the repository.

## Verification

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\verify.ps1'
```

Acceptance requires backend lint and tests, archive/catalogue/model verification, frontend lint and typecheck, and a production frontend build to pass.

## Presentation flow

1. Open Dashboard to show prepared scenarios, model readiness, and saved-run totals.
2. Open Analyse, select a preloaded scenario, and run local inference.
3. Compare the AI overlay with dataset ground truth.
4. Open Analytics to show real occupancy and quality metrics.
5. Open Reports and download the run as JSON or CSV.

No hardware, live feed, cloud inference API, or network connection is required during the demo.
