# Milestone 5 — Guided Presentation Mode and Demo Readiness

Milestone 5 provides a deterministic, presentation-friendly way to demonstrate the complete offline product without manually selecting inputs during the live session.

## Included

- Five-point preflight for the prepared catalogue, three-dataset coverage, local images, AI model, and SQLite.
- Stable selection of one information-rich scenario from PKLot, CNRPark+EXT, and ACPDS.
- Guided sequential inference using the existing local model and normal history persistence.
- Full-screen presentation control, run progress, dataset result tabs, overlays, AI quality metrics, and detailed report links.
- Dashboard and sidebar access to the new Presentation page.
- Unit tests for deterministic scenario selection and readiness rules.
- Command-line demo preflight integrated into `scripts\verify.ps1`.

## API additions

- `GET /api/v1/demo/readiness`
- `GET /api/v1/demo/showcase`

The existing `POST /api/v1/analysis/scenarios/{scenario_id}` endpoint performs each showcase inference. Presentation runs remain visible in History, Analytics, and Reports.

## Demonstration

1. Run `scripts\verify.ps1` before the presentation.
2. Start the backend and frontend.
3. Open `http://localhost:3000/presentation`.
4. Confirm all five checks are ready.
5. Select **Run guided showcase**.
6. Present each completed dataset from the result tabs.
7. Open Analytics to compare the three newly saved runs.

The showcase uses only preloaded data and the existing local model. No hardware, live feed, network access, or cloud AI is required.
