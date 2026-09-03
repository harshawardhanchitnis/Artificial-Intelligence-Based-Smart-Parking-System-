# Milestone 7 — Application Hardening and End-to-End Reliability

Milestone 7 hardens the complete offline product from the corrected baseline commit
`b4cb4813d9572d580d914df85583707c75df8947`. It does not retrain, retune, or modify the
leakage-safe benchmark or model methodology.

## Backend reliability

- Separate liveness and deep readiness endpoints.
- Seven readiness checks covering SQLite integrity, datasets, catalogue, showcase media, local AI,
  and independent benchmark metadata.
- Stable error envelopes for HTTP and request-validation failures.
- Request-ID propagation and correlation for browser-visible failures.
- Processing-time and defensive browser headers on every API response.
- Safe generic handling and logging for unexpected server errors.
- SQLite foreign-key enforcement, 30-second busy timeout, write-ahead logging, and connection
  pre-ping.

## Frontend resilience

- One typed API client for all JSON requests.
- Bounded request timeouts with longer limits only for local inference.
- Useful API error messages including request IDs when available.
- Retry controls for failed read-only screens.
- Header status based on actual deep readiness rather than a hard-coded ready label.
- System page exposes every readiness dependency and supports manual rechecking.

## End-to-end verification

`scripts/reliability-check.ps1` checks:

1. backend liveness and deep readiness;
2. catalogue, scenarios, model, dashboard, history, analytics, diagnostics, and showcase APIs;
3. the unchanged 1,800-sample unseen benchmark contract;
4. all nine frontend routes; and
5. optionally, one real inference plus SQLite persistence.

The default run is read-only. `-IncludeInference` deliberately creates one analysis-history row.

## Commands

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\setup.ps1'
powershell -ExecutionPolicy Bypass -File '.\scripts\verify.ps1'
```

After starting both services:

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\reliability-check.ps1'
powershell -ExecutionPolicy Bypass -File '.\scripts\reliability-check.ps1' -IncludeInference
```

## Preserved boundaries

- No hardware, sensors, cameras, RTSP feeds, live traffic data, or cloud AI APIs.
- No training or benchmark changes.
- No datasets, benchmark cache, model weights, databases, dependencies, environments, generated
  media, reports, or secrets committed.
- `Technical Documents/` remains untouched and untracked.
