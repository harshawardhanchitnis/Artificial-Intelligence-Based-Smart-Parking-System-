# Version 1.0 presentation runbook

This is the definitive final demonstration path. Every input is already stored locally. Do not
connect a camera, upload a new image, retrain the model, or depend on internet access during the
presentation.

## Before presentation day

1. Run `scripts\verify.ps1` and retain the successful terminal output.
2. Start both services and run `scripts\release-check.ps1 -IncludeInference` once.
3. Confirm the System page shows **Version 1.0**, **final**, and 7/7 readiness checks.
4. Confirm Presentation shows five green checks and three showcase scenarios.
5. Keep the laptop connected to power and disable disruptive notifications and sleep.

## Fifteen-minute preflight

Open two PowerShell terminals in the repository root:

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\start-backend.ps1'
```

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\start-frontend.ps1'
```

Then run the read-only handover check in a third terminal:

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\release-check.ps1'
```

Open <http://localhost:3000/presentation>. Keep
<http://localhost:3000/system> open in a second browser tab.

## Five-minute talk track

| Time | Screen | Demonstration |
| --- | --- | --- |
| 0:00–0:35 | System | Identify Version 1.0, offline mode, 7/7 readiness, and the local model. |
| 0:35–2:40 | Presentation | Start the guided showcase. Analyse PKLot, CNRPark+EXT, and ACPDS in the fixed order. Explain red occupied and green vacant slots, totals, confidence, and processing time. |
| 2:40–3:25 | Presentation results | Switch dataset tabs and compare AI predictions with verified ground truth. Emphasize that each run is saved locally. |
| 3:25–4:10 | Diagnostics | Show the independent 1,800-sample unseen-test result separately from stored-run agreement. Explain the confusion matrix and dataset breakdown. |
| 4:10–4:40 | History / Analytics | Open a newly saved run and show the detailed overlay, then show aggregate local usage. |
| 4:40–5:00 | Reports | Download or open a JSON/CSV report and close with the no-hardware, no-live-data, no-cloud-AI boundary. |

## Key statements

- “The product uses preloaded images from PKLot, CNRPark+EXT, and ACPDS.”
- “The model runs locally and classifies each predefined parking space.”
- “Green is vacant and red is occupied; the counters are derived from these slot predictions.”
- “The verified unseen-test accuracy is 90.7% on 1,800 balanced, independent samples.”
- “Saved-run agreement is operational evidence, not a replacement for the independent benchmark.”

## Recovery during the presentation

| Symptom | Safe response |
| --- | --- |
| Header says backend offline | Restart `scripts\start-backend.ps1`, wait for `/health/live`, then press Retry. |
| Readiness shows attention | Open System, read the failed check, and restore that local dependency. Do not retrain. |
| Frontend page does not load | Restart `scripts\start-frontend.ps1` and refresh; backend history remains intact. |
| One inference fails | Keep the prepared result tabs visible, restart the backend, and retry the scenario once. |
| Internet is unavailable | Continue normally; Version 1.0 requires no network connection. |

Do not delete the SQLite database, dataset root, benchmark cache, or model artifacts as a recovery
step. Do not run training or dataset preparation immediately before or during the presentation.
