# Milestone 2 — Offline Dataset Pipeline and Scenario Catalogue

Milestone 2 prepares a small presentation-safe catalogue from PKLot, CNRPark+EXT, and ACPDS without copying any source datasets into Git. It normalizes each dataset's annotations to polygons with occupied/vacant ground-truth states and exposes the catalogue through FastAPI and the frontend.

## Branch preparation

Run from the repository root while `milestone/01-foundation` is clean and up to date:

```powershell
git switch milestone/01-foundation
git pull --ff-only
git switch -c milestone/02-dataset-pipeline
```

Apply either the ZIP or the patch supplied with this milestone, never both.

## Setup and preparation

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\setup.ps1'
powershell -ExecutionPolicy Bypass -File '.\scripts\prepare-datasets.ps1' -Profile Plan
powershell -ExecutionPolicy Bypass -File '.\scripts\prepare-datasets.ps1' -Profile Demo -SamplesPerDataset 3
```

The Demo profile streams selected images and annotations directly from the archives. It creates only small external files under `PARKING_DATA_ROOT\demo` and `PARKING_DATA_ROOT\prepared`. Re-running it is safe. Use `-Force` only when you intentionally want to replace prepared copies.

Full extraction is optional, is not required for the application or presentation, and is blocked unless explicitly confirmed:

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\prepare-datasets.ps1' -Profile Full -ConfirmFullExtraction
```

## Verification

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\verify.ps1'
```

Acceptance criteria:

- all backend tests and lint checks pass;
- all seven source archives validate;
- the prepared-data verifier reports `valid: true`;
- the catalogue contains PKLot, CNRPark+EXT, and ACPDS scenarios;
- frontend lint, typecheck, and production build pass;
- `/api/v1/datasets/catalogue` reports `prepared: true`;
- `/api/v1/datasets/scenarios` returns normalized scenarios;
- Analyse displays the selected source image, green vacant polygons, red occupied polygons, and correct totals;
- Parking Lots displays catalogue counts, lots/cameras, and conditions;
- System displays `Demo catalogue: Prepared`.

## Commit and push

Inspect the changes first and confirm no prohibited files are staged:

```powershell
git status --short
git diff --check
git add .github .vscode backend data docs frontend generated ml models scripts .editorconfig .env.example .gitattributes .gitignore docker-compose.yml README.md MILESTONE-01.md MILESTONE-02.md
git status --short
git diff --cached --stat
git commit -m "feat: add offline dataset preparation and scenario catalogue"
git push -u origin milestone/02-dataset-pipeline
git status
```

Do not add `Technical Documents/`, `.env`, dataset archives, raw or prepared datasets, generated media, environments, dependencies, runtime databases, model weights, or secrets.
