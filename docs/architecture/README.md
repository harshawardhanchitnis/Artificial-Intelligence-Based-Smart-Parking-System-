# Foundation architecture

The product is a local two-process application:

1. The Next.js frontend presents the dashboard, catalogue, analysis workspace, history, analytics, reports, and system readiness pages.
2. The FastAPI backend owns dataset access, AI inference, occupancy decisions, generated overlays, persistence, and reports.
3. SQLite stores prepared scenarios and analysis history.
4. Original datasets remain outside Git and are accessed through a configured read-only path.
5. Model weights and generated media remain local.

## Planned request flow

```text
User selects scenario
        ↓
Frontend requests analysis
        ↓
FastAPI loads image + slot geometry
        ↓
Local classifier determines each slot state
        ↓
Backend renders green/red overlay
        ↓
SQLite stores summary and output path
        ↓
Frontend presents results
```

## Boundaries

There are no hardware adapters, live streams, external inference APIs, research experiments, reservation payments, or navigation services in the first product version.
