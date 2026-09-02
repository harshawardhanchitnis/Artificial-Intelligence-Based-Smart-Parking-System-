# Presentation plan

## Milestone 4 demonstration

Before presenting, run `scripts\verify.ps1` and confirm the model is ready on the System page. Analyse one scenario from each configured dataset so the Analytics comparison has representative rows. The recommended live sequence is Dashboard, Analyse, Analytics, History, then Reports. Download an individual JSON report to show that each visible parking slot has a saved prediction, confidence, and ground-truth comparison.

All demonstration inputs are already prepared locally. No camera, sensor, cloud AI service, or internet connection is used.

The final demonstration is designed to remain reliable without internet access.

1. Start the backend and frontend.
2. Open System and confirm database, dataset catalogue, and local model readiness.
3. Open Analyse.
4. Select PKLot, CNRPark+EXT, or ACPDS.
5. Select a prepared lot, condition, and scenario.
6. Show the verified dataset ground-truth overlay.
7. Click **Run local AI analysis**.
8. Explain green vacant and red occupied predictions, confidence, processing time, and ground-truth agreement.
9. Switch between Ground truth and AI prediction to demonstrate the comparison.
10. Open History and show the newly stored SQLite analysis record.

Prepare the curated catalogue before the presentation. No user upload, internet connection, live feed, camera, or hardware is required.
