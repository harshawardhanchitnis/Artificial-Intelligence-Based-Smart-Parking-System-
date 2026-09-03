# Baseline ML Validation and Evaluation Correction

This focused correction sits between Milestone 6 and Milestone 7. It preserves every product
route while replacing the 426-slot development fit with a reproducible, leakage-safe benchmark.

## Required base

- Branch: `milestone/06-ai-diagnostics`
- Commit: `b0c2e5f117322a4a81fe88e03c7a79aa3f59b7cc`

## Root cause

Milestone 3 evaluated a scenario holdout, selected a threshold, and then refitted the deployed
model on all 426 demo-catalogue slots. Milestone 6 correctly counted stored predictions, but
repeated UI and Presentation executions of those already-fitted scenarios increased the apparent
sample count. Those values are stored-run agreement, not independent model accuracy.

The audited local database contained 12 prediction-enabled runs, 875 stored predictions,
339 unique scenario/slot pairs, and five unique scenarios. Repetition inflated the displayed
count by 2.58 times.

## Corrected protocol

- Training: 6,000 balanced patches, 2,000 from each approved dataset.
- Validation: 1,800 balanced, group-isolated patches used only to select the threshold.
- Unseen test: 1,800 balanced, group-isolated patches used only for final evaluation.
- PKLot groups are site/date combinations.
- CNRPark+EXT keeps official test days and reserves three complete training days for validation.
- ACPDS keeps official train/valid/test image partitions.
- Source-image, group, and duplicate-content overlap are rejected automatically.
- The deployed v2 model is fitted on training only; it is not refitted on validation or test.

## Verified benchmark

The default seed 42 profile produced:

- validation accuracy and balanced accuracy: 93.33%;
- unseen-test accuracy and balanced accuracy: 90.67%;
- unseen occupied precision: 92.66%;
- unseen occupied recall: 88.33%;
- unseen occupied F1: 90.44%;
- unseen vacant specificity: 93.00%;
- confusion matrix: TN 837, FP 63, FN 105, TP 795;
- dataset accuracy: ACPDS 85.67%, CNRPark+EXT 91.33%, PKLot 95.00%.

The benchmark manifest SHA-256 is
`672ee5905f6ed25e11dc74c54a7356cf857d43c7d992925a84eaea16d1930539`.

## Prepare, train, and benchmark

```powershell
powershell -ExecutionPolicy Bypass -File '.\scripts\setup.ps1'
powershell -ExecutionPolicy Bypass -File '.\scripts\prepare-datasets.ps1' -Profile Benchmark
powershell -ExecutionPolicy Bypass -File '.\scripts\train-model.ps1'
powershell -ExecutionPolicy Bypass -File '.\scripts\benchmark-model.ps1'
powershell -ExecutionPolicy Bypass -File '.\scripts\verify.ps1'
```

The first Benchmark preparation scans the compressed PKLot archive twice and can take several
minutes. It does not perform full dataset extraction.

## Diagnostics semantics

The page now has two explicit sections:

1. **Independent model benchmark** — validation, pooled unseen test, per-dataset unseen test,
   confusion matrices, and unique sample/source/group counts.
2. **Application run diagnostics** — stored-run agreement, prediction-enabled and legacy runs,
   stored predictions, unique scenarios, unique scenario slots, and repeated predictions.

Running the same presentation scenario again changes only the second section.

## Repository policy

Do not stage `Technical Documents/`, the benchmark cache, raw archives, prepared data, SQLite
databases, environments, dependencies, generated model files, generated media, outputs, or
secrets.
