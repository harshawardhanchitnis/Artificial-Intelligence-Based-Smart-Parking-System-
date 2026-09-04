# ParkSense AI — Version 1.0

Version 1.0 is the final offline, dataset-driven smart-parking product prepared for demonstration
and handover. It analyses curated parking-lot images, classifies every defined parking space as
occupied or vacant, visualizes the result, and preserves auditable local history.

## Product capabilities

- Prepared scenarios from PKLot, CNRPark+EXT, and ACPDS.
- Local per-slot occupancy inference without an internet connection or external AI API.
- Prediction and verified ground-truth overlays with occupancy totals and confidence.
- SQLite-backed history, analytics, diagnostics, and CSV/JSON reporting.
- Deterministic three-dataset presentation mode.
- Deep readiness, request tracing, safe error handling, retry/recovery UI, and repeatable runtime
  reliability checks.

## Verified AI baseline

| Property | Version 1.0 contract |
| --- | ---: |
| Model | `parking-occupancy-logistic-v2` |
| Training partition | 6,000 balanced samples |
| Validation partition | 1,800 balanced samples |
| Unseen-test partition | 1,800 balanced samples |
| Decision threshold | 0.55 |
| Unseen-test accuracy | 90.7% |
| Balanced accuracy | 90.7% |
| Occupied precision | 92.7% |
| Occupied recall | 88.3% |
| Occupied F1 | 90.4% |
| Vacant specificity | 93.0% |

The unseen-test figures are an independent benchmark. Repeated saved application runs are shown
separately as stored-run agreement and are not presented as a generalization estimate.

## Operating boundaries

Version 1.0 intentionally uses no hardware, cameras, sensors, RTSP streams, live traffic data,
Gemini/OpenAI inference APIs, or other cloud AI services. The prepared data, model artifacts,
database, and generated reports remain local and are excluded from Git.
