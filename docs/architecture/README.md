# Foundation architecture

## Baseline ML validation correction

`Source archives → deterministic grouped manifest → train-only fit → validation threshold → unseen test benchmark`

The v2 local classifier is fitted only on the Benchmark profile's training partition. Validation
selects the decision threshold, while unseen test samples remain excluded from every fitting and
tuning decision. The Diagnostics page presents this independent benchmark separately from
repeated application-run agreement. The eight-scenario demo catalogue remains the source of
interactive presentation images and is not used as the scientific benchmark.

## Milestone 6 diagnostic flow

`Saved prediction JSON → confusion classification → overall/dataset metrics → diagnostics UI → analysis detail inspection`

Diagnostics are derived at read time from the existing SQLite analysis records. No duplicate model run, diagnostic database, or generated metric file is introduced. Legacy rows without per-slot predictions remain visible but are explicitly excluded from slot-level calculations.

## Milestone 5 presentation flow

`Preflight → deterministic showcase catalogue → three existing analysis requests → SQLite history → guided overlays → Analytics/Reports`

Presentation mode does not introduce a second inference path. It selects one stable scenario per approved dataset and calls the same analysis endpoint used by the normal workspace, preserving consistent AI behavior and audit history.

## Milestone 4 data flow

`Prepared scenario → local inference → AnalysisRecord → dashboard/analytics/report APIs → Next.js views`

SQLite remains the only application database. The version 4 migration adds nullable model-quality and prediction fields to the existing analysis table, allowing older rows to remain visible as legacy runs. CSV and JSON exports are streamed directly from API responses and do not create runtime files in Git.

The product is a local two-process application:

1. The Next.js frontend presents the dashboard, catalogue, analysis workspace, history, analytics, reports, and system readiness pages.
2. The FastAPI backend owns dataset access, AI inference, occupancy decisions, generated overlays, persistence, and reports.
3. The external `demo/catalogue.json` stores prepared scenarios; SQLite stores application and analysis history records.
4. Original datasets remain outside Git and are accessed through a configured read-only path.
5. Model weights and generated media remain local.

## Milestone 3 request flow

```text
User selects scenario
        ↓
Frontend requests local AI analysis
        ↓
FastAPI loads image, slot geometry, and verified local model
        ↓
Feature extractor builds HOG/color vectors for every slot crop
        ↓
Local classifier predicts occupied probability and confidence
        ↓
SQLite records prediction totals and processing time
        ↓
Frontend renders prediction polygons and ground-truth comparison
        ↓
Frontend presents results
```

The catalogue remains unchanged. Numeric model weights and metadata live under the external data root, and a SHA-256 check is required before inference. No pickle, cloud model, camera, sensor, or live feed is involved.

## Milestone 7 reliability boundary

Milestone 7 keeps liveness separate from readiness. `/api/v1/health/live` confirms that the API
process can respond. `/api/v1/health/ready` performs seven local dependency checks and returns
HTTP 503 when the product is not presentation-ready. The checks cover the external data root,
SQLite `quick_check`, prepared catalogue, three-dataset coverage, showcase images, the local model,
and the immutable unseen benchmark metadata.

Every API response includes a validated request ID, elapsed processing time, and defensive browser
headers. Expected HTTP and validation failures use a stable JSON error envelope; unexpected failures
are logged with their request ID without exposing a traceback to the browser. SQLite uses a 30-second
busy timeout, foreign-key enforcement, write-ahead logging, and connection pre-ping for safer local
concurrency.

## Boundaries

There are no hardware adapters, live streams, external inference APIs, research experiments, reservation payments, or navigation services in the first product version.
