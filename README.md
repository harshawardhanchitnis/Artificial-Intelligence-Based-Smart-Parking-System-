# Artificial Intelligence Based Smart Parking System

An offline, dataset-driven smart-parking application. The product analyses preloaded parking-lot images and presents occupied and vacant spaces without cameras, sensors, live feeds, or cloud AI APIs.

## Baseline ML validation correction status

The application now includes:

- Next.js, React, TypeScript, and Tailwind CSS frontend shell
- FastAPI backend with health, system, and dataset endpoints
- SQLite initialization and an analysis-history model
- PKLot, CNRPark+EXT, and ACPDS archive validation
- Windows setup, start, and verification scripts
- CI, VS Code settings, architecture notes, and presentation plan
- Safe, streaming adapters for PKLot, CNRPark+EXT, and ACPDS
- A lightweight Demo profile with normalized slot polygons and occupancy labels
- Prepared catalogue/scenario/image APIs
- Interactive Analyse and Parking Lots pages backed by real prepared data
- Local occupancy model training with scenario-group validation
- Checksum-verified, non-pickle numeric model persistence
- Per-slot AI predictions, confidence, comparison overlays, and SQLite history
- Safe additive migration for existing analysis-history databases
- Live dashboard and operational analytics from saved local inference runs
- On-demand full-history CSV and per-analysis CSV/JSON reports
- Five-point offline presentation readiness preflight
- Deterministic three-dataset showcase with sequential local inference
- Full-screen guided results with overlays, quality metrics, and report links
- Confusion-matrix diagnostics with accuracy, precision, recall, specificity, and F1
- Dataset-level AI quality comparison and recent error inspection queue
- Detailed saved-analysis pages with prediction/ground-truth and errors-only overlays
- Reproducible, group-safe train/validation/unseen-test benchmark preparation
- Independent benchmark metrics separated from repeated application-run agreement
- Deep seven-point readiness checks with SQLite integrity validation
- Request IDs, safe API errors, response timing, and browser security headers
- Frontend API timeouts, retry states, and a live local-system status indicator
- Repeatable nine-route runtime smoke testing with optional persisted inference

The application now distinguishes verified dataset ground truth from actual local AI predictions, persists model-quality fields, and turns saved runs into presentation analytics and reports. The lightweight model runs on CPU, requires no pretrained download, and remains fully offline. Full dataset extraction is still optional and is not required for the demo.

## Prerequisites

- Windows 10 or 11
- Git
- Python 3.13
- Node.js 22 and npm
- Dataset archives stored in `D:\Projects\AI Based Smart Parking System Data\archives`

## First-time setup

From PowerShell in the repository root:

```powershell
Copy-Item -LiteralPath '.env.example' -Destination '.env'
powershell -ExecutionPolicy Bypass -File '.\scripts\setup.ps1'
```

The setup script creates `backend\.venv`, installs the backend and frontend dependencies, initializes SQLite, validates the dataset archive catalogue, and runs smoke tests.

Existing Milestone 3 databases are upgraded safely during setup. The migration can also be run directly with `scripts\migrate-database.ps1`.

Prepare the lightweight offline catalogue and scientific benchmark:

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\prepare-datasets.ps1' -Profile Plan
powershell -ExecutionPolicy Bypass -File '.\scripts\prepare-datasets.ps1' -Profile Demo -SamplesPerDataset 3
powershell -ExecutionPolicy Bypass -File '.\scripts\prepare-datasets.ps1' -Profile Benchmark
powershell -ExecutionPolicy Bypass -File '.\scripts\train-model.ps1'
powershell -ExecutionPolicy Bypass -File '.\scripts\benchmark-model.ps1'
```

## Start the application

Open two VS Code terminals.

Terminal 1:

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\start-backend.ps1'
```

Terminal 2:

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\start-frontend.ps1'
```

Then open:

- Frontend: <http://localhost:3000>
- Backend API documentation: <http://127.0.0.1:8000/docs>
- Backend health: <http://127.0.0.1:8000/api/v1/health>

## Verification

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\verify.ps1'
```

With the backend and frontend running, verify the complete local product without changing
history:

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\reliability-check.ps1'
```

To include one real inference and persistence check against the current local database, add
`-IncludeInference`. This intentionally creates one new analysis-history row.

## Dataset policy

The original datasets, extracted images, generated demo media, SQLite databases, and trained model weights are not committed to Git. The external dataset root is configured with `PARKING_DATA_ROOT` in `.env`.

The bounded benchmark cache is written under `PARKING_DATA_ROOT\prepared\ml-benchmark`.
Its manifest, selected patches, reports, and model artifacts remain local and outside Git.

## Project boundaries

- No hardware or sensor integration
- No CCTV, webcam, RTSP, or live traffic data
- No Gemini, OpenAI, or other inference API
- The final presentation uses curated, preloaded scenarios
- AI inference runs locally and produces slot overlays and occupancy totals
