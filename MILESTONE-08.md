# Milestone 8 — Final Version 1.0 and Presentation Handover

Milestone 8 promotes the verified Milestone 7 product to a clearly identified, reproducible
Version 1.0 release candidate. It is based exactly on
`2092239e30be6ba17c4686e02254785b1c7205be` and does not retrain, retune, regenerate, or
otherwise modify the protected model or benchmark.

## Version 1.0 release contract

- One canonical `VERSION` file and matching backend/frontend package versions.
- FastAPI, liveness, system, and release endpoints report Version 1.0 consistently.
- `/api/v1/system/release` exposes the approved offline datasets, capabilities, boundaries, and
  immutable model/evaluation contract.
- The System page visibly identifies the final release and its verified unseen-test result.
- A seven-check release gate rejects version, readiness, dataset, model, threshold, partition, or
  benchmark drift.

## Handover verification

Run the complete static verification before starting the services:

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\verify.ps1'
```

With the backend and frontend running, validate the release and presentation contracts:

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\release-check.ps1'
```

For one final end-to-end inference and SQLite persistence check, add `-IncludeInference`. This
option deliberately adds one analysis-history row; the default release check is read-only.

## Handover package

- `RELEASE-NOTES.md` records the Version 1.0 product scope and verified ML figures.
- `docs/release/README.md` contains the release, rollback, and optional post-acceptance Git steps.
- `docs/presentation-guide/README.md` is the definitive offline presentation runbook, including a
  five-minute talk track and recovery paths.

## Preserved boundaries

- No hardware, camera, sensor, live feed, traffic service, or cloud AI integration.
- No changes to the leakage-safe 6,000/1,800/1,800 train/validation/unseen-test split.
- No changes to `parking-occupancy-logistic-v2`, its 0.55 threshold, or its verified 90.7%
  unseen-test accuracy.
- No datasets, archives, benchmark cache, model weights, SQLite databases, environments,
  dependencies, generated media, reports, or secrets committed.
- `Technical Documents/` remains untouched and untracked.
