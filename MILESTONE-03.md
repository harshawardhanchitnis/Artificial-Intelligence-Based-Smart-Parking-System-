# Milestone 3 — Local AI Occupancy Inference

Milestone 3 trains and runs a compact local machine-learning classifier on the prepared Milestone 2 catalogue. It predicts each parking space as occupied or vacant, exposes confidence and ground-truth agreement, stores analysis history in SQLite, and keeps the verified catalogue unchanged.

## Branch preparation

```powershell
Set-Location -LiteralPath 'D:\Projects\AI Based Smart Parking System'
git switch milestone/02-dataset-pipeline
git pull --ff-only
git rev-parse HEAD
git switch -c milestone/03-local-ai-inference
```

The required base commit is `381bbfb8838c43940810d1b0348892abc7b98c46`.

## Setup and training

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\setup.ps1'
powershell -ExecutionPolicy Bypass -File '.\scripts\train-model.ps1'
powershell -ExecutionPolicy Bypass -File '.\scripts\verify.ps1'
```

No pretrained model or API key is required. Setup downloads pinned Python packages from PyPI. Training reads the existing external `demo/catalogue.json` and images, then writes these local artifacts under `PARKING_DATA_ROOT\models`:

- `parking-occupancy-logistic-v1.npz` — numeric coefficients only;
- `parking-occupancy-logistic-v1.json` — metadata, metrics, versions, and SHA-256.

The application loads NumPy arrays with `allow_pickle=False` and verifies the weight checksum before inference. Model artifacts are generated locally and must not be committed.

## Model design

- Slot crops come from the normalized Milestone 2 polygons.
- Deterministic HOG, color-histogram, color-statistic, and grayscale-thumbnail features produce 238 values per slot.
- A standardized, class-balanced logistic regression learns occupied versus vacant states.
- Validation splits complete scenarios, not random slots, reducing same-image leakage.
- The validation threshold is selected only on the scenario holdout; the final presentation model is then fitted to all prepared slots.
- Inference is CPU-first and offline. The RTX GPU is not required for this lightweight baseline.

## API additions

- `GET /api/v1/model/status`
- `POST /api/v1/analysis/scenarios/{scenario_id}`
- `GET /api/v1/analysis/history`

## Acceptance criteria

- backend formatting, lint, and tests pass;
- all source archives and prepared data still verify;
- training reports `ready: true` and covers all three datasets;
- model metadata and weights exist only under the external data root;
- model verification reports `valid: true` for all prepared scenarios and slots;
- model/status reports `ready: true`;
- a POST analysis returns predictions, confidence, totals, agreement, and processing time;
- Analyse switches between ground truth and AI prediction overlays;
- History displays the SQLite record created by an analysis;
- System reports `Local AI model: Ready`;
- frontend lint, typecheck, and production build pass;
- no cloud AI, live data, camera, sensor, or pretrained model is used.

## Commit and push

```powershell
git add .github .vscode backend data docs frontend generated ml models scripts .editorconfig .env.example .gitattributes .gitignore docker-compose.yml README.md MILESTONE-01.md MILESTONE-02.md MILESTONE-03.md
git status --short
git diff --cached --check
git diff --cached --stat
git commit -m "feat: add local AI occupancy inference pipeline"
git push -u origin milestone/03-local-ai-inference
git status
```

Do not stage `Technical Documents/`, `.env`, datasets, demo media, runtime databases, environments, dependencies, model artifacts, generated outputs, or secrets.
