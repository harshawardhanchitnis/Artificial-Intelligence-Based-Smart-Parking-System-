# Artificial Intelligence Based Smart Parking System

An offline, dataset-driven smart-parking application. The product analyses preloaded parking-lot images and presents occupied and vacant spaces without cameras, sensors, live feeds, or cloud AI APIs.

## Milestone 3 status

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

The application now distinguishes verified dataset ground truth from actual local AI predictions. The lightweight model runs on CPU, requires no pretrained download, and remains fully offline. Full dataset extraction is still optional and is not required for the demo.

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

Prepare the lightweight offline catalogue:

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\prepare-datasets.ps1' -Profile Plan
powershell -ExecutionPolicy Bypass -File '.\scripts\prepare-datasets.ps1' -Profile Demo -SamplesPerDataset 3
powershell -ExecutionPolicy Bypass -File '.\scripts\train-model.ps1'
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

## Dataset policy

The original datasets, extracted images, generated demo media, SQLite databases, and trained model weights are not committed to Git. The external dataset root is configured with `PARKING_DATA_ROOT` in `.env`.

## Project boundaries

- No hardware or sensor integration
- No CCTV, webcam, RTSP, or live traffic data
- No Gemini, OpenAI, or other inference API
- The final presentation uses curated, preloaded scenarios
- AI inference runs locally and produces slot overlays and occupancy totals
