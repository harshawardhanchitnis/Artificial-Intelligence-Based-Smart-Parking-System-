# Version 1.0 release handover

## Release identity

- Product: ParkSense AI — Artificial Intelligence Based Smart Parking System
- Version: `1.0.0`
- Release stage: `final`
- Runtime mode: offline
- Required datasets: PKLot, CNRPark+EXT, and ACPDS
- Required model: `parking-occupancy-logistic-v2`

`VERSION`, the backend and frontend package versions, the FastAPI document version, and the
release API must all agree. `python -m app.cli.verify_release` enforces the complete model and
evaluation contract as well as the seven-point application readiness gate.

## Final local acceptance

1. Run `scripts\setup.ps1` when installing on a new machine or environment.
2. Run `scripts\verify.ps1` with both application services stopped.
3. Start the backend and frontend in separate terminals.
4. Run `scripts\release-check.ps1` for a read-only runtime check.
5. Run `scripts\release-check.ps1 -IncludeInference` once if final persistence verification is
   required. This adds one history record.
6. Follow `docs\presentation-guide\README.md` for the final demonstration.

Acceptance requires all backend tests, frontend checks, archive/catalogue/model/benchmark
validation, deep readiness, nine frontend routes, the Version 1.0 release contract, and the
three-dataset presentation preflight to pass.

## Recovery and rollback

- If the backend is unavailable, restart `scripts\start-backend.ps1` and use the UI retry control.
- If the frontend is unavailable, restart `scripts\start-frontend.ps1`; the backend and stored data
  are unaffected.
- If readiness fails, open System and address the named dependency instead of retraining.
- If a release change must be rolled back, switch back to the verified Milestone 7 commit
  `2092239e30be6ba17c4686e02254785b1c7205be`. Do not delete the external data root.

## Optional Git release after acceptance

Milestone delivery intentionally does not merge, tag, or change the default branch. After the
Milestone 8 branch is independently accepted, the repository owner may merge it and create the
annotated release tag:

```powershell
git switch main
git merge --ff-only milestone/08-final-v1-handover
git tag -a v1.0.0 -m 'ParkSense AI Version 1.0'
git push origin main
git push origin v1.0.0
```

Run these commands only after confirming the intended main branch and release policy on GitHub.
