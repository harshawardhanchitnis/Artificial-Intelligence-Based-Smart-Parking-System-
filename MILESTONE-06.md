# Milestone 6 — AI Quality Diagnostics and Analysis Detail

Milestone 6 turns stored per-slot predictions into product-level quality diagnostics and inspectable analysis records. It uses verified dataset ground truth already saved with current inference runs.

## Included

- Aggregate confusion matrix across prediction-enabled analyses.
- Accuracy, occupied precision, occupied recall, vacant specificity, and F1 score.
- Dataset-level quality comparison for PKLot, CNRPark+EXT, and ACPDS.
- Explicit coverage counts for detailed and legacy analysis records.
- Recent incorrect-slot inspection queue.
- Detailed analysis API and browser page with prediction/ground-truth overlay switching.
- Errors-only overlay filtering with yellow incorrect-slot borders.
- Direct detail links from History and JSON report download from each analysis.
- Unit tests for metric calculation, error classification, and empty/legacy handling.

## API additions

- `GET /api/v1/diagnostics/summary`
- `GET /api/v1/analysis/history/{analysis_id}`

## Product interpretation

- **Accuracy**: all correct slot decisions divided by evaluated slots.
- **Precision**: correctly predicted occupied slots divided by all occupied predictions.
- **Recall**: correctly predicted occupied slots divided by all truly occupied slots.
- **Specificity**: correctly predicted vacant slots divided by all truly vacant slots.
- **F1**: harmonic balance of occupied precision and recall.

Legacy runs remain in History but are excluded from per-slot diagnostics because they predate saved predictions.

## Demonstration

Run the guided showcase, open **AI diagnostics**, and explain the confusion matrix and dataset comparison. Select an inspection-queue item or an **Inspect** action in History to compare the saved AI overlay with verified ground truth.

No model retraining, dataset preparation, hardware, live feed, or cloud API is required.
