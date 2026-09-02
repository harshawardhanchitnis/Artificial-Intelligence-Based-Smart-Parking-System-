# Foundation architecture

The product is a local two-process application:

1. The Next.js frontend presents the dashboard, catalogue, analysis workspace, history, analytics, reports, and system readiness pages.
2. The FastAPI backend owns dataset access, AI inference, occupancy decisions, generated overlays, persistence, and reports.
3. The external `demo/catalogue.json` stores prepared scenarios; SQLite stores application and analysis history records.
4. Original datasets remain outside Git and are accessed through a configured read-only path.
5. Model weights and generated media remain local.

## Milestone 2 request flow

```text
User selects scenario
        ↓
Frontend requests prepared scenario
        ↓
FastAPI loads catalogue + normalized slot geometry
        ↓
Dataset ground truth supplies each slot state
        ↓
Frontend renders green/red polygons over the source image
        ↓
Frontend presents results
```

Milestone 3 inserts local model inference between image loading and result presentation. The catalogue contract remains stable so the UI can compare ground truth with predictions later.

## Boundaries

There are no hardware adapters, live streams, external inference APIs, research experiments, reservation payments, or navigation services in the first product version.
