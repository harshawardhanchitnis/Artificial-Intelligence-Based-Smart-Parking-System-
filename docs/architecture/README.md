# Foundation architecture

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

## Boundaries

There are no hardware adapters, live streams, external inference APIs, research experiments, reservation payments, or navigation services in the first product version.
