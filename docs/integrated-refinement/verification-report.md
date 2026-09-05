# Integrated refinement verification report

## Scope and provenance

- Required frozen base: `2245350771a94088be41b9b51093686acbbbd06d`
- Development branch: `refinement/integrated-product-foundation`
- The frozen `milestone/08-final-v1-handover` branch was not changed.
- `Technical Documents/` was not read, changed, staged, or packaged.
- Dataset archives, prepared media, databases, environments, dependencies, secrets, checkpoints,
  and model weights remain outside Git.

## Protected evaluation results

Both decision locks were written before their respective holdouts were opened. Each final holdout
was evaluated once; the finalization commands refuse a repeat.

### Occupancy classifier

`parking-occupancy-enhanced-v3` is a MobileNetV3-Small slot classifier exported to ONNX. The
historical `parking-occupancy-logistic-v2` remains available as an unchanged reproducible baseline.

| Metric | Final protected holdout |
|---|---:|
| Samples | 25,113 |
| Accuracy | 97.906% |
| Balanced accuracy | 97.906% |
| Occupied precision | 96.469% |
| Occupied recall | 99.450% |
| Occupied F1 | 97.937% |
| Vacant specificity | 96.361% |
| False-vacant rate | 0.550% |
| False-occupied rate | 3.639% |

Per-dataset balanced accuracy was 94.630% for ACPDS, 99.730% for CNRPark+EXT, and 96.442% for
PKLot. The weak PUCPR/2012-10-30 group is disclosed in the frozen report instead of being hidden or
used for post-holdout tuning. The 5.806 MB ONNX model measured 1.346 ms per slot for ONNX execution
and 2.833 ms per slot including local preprocessing on the verification computer.

### Automatic fixed-camera space localisation

The deployed strategy classifies one of the twelve validated PKLot/CNRPark+EXT fixed-camera
layouts and returns its verified canonical polygons. An aspect-family constraint prevents 16:9
PKLot images from being confused with 4:3 CNR views. A neural polygon detector remains an explicitly
diagnostic artifact because it did not meet the validation gate.

| Metric | Validation | Final protected holdout |
|---|---:|---:|
| Images | 600 | 710 |
| Precision at IoU 0.50 | 94.102% | 93.537% |
| Recall at IoU 0.50 | 90.722% | 88.150% |
| F1 at IoU 0.50 | 92.381% | 90.764% |
| Recall at IoU 0.75 | 90.465% | 87.693% |

Final PKLot precision/recall was 90.332%/81.714%; CNRPark+EXT was 98.036%/98.148%. Occupancy on
29,525 geometry-matched detected slots achieved 97.358% accuracy and 96.934% balanced accuracy.
The measured localizer step was 26.679 ms per image; complete CPU image analysis averaged 2.334 s
with a 2.427 s p95. ACPDS moving-camera GoPro frames are deliberately outside this automatic
fixed-camera localisation claim and remain supported through verified prepared geometry.

## Data and regression evidence

- Leakage-safe protocol: 106,907 crops from 3,224 source images and 561 groups.
- Partitions: 65,805 train, 15,989 validation, and 25,113 final protected holdout crops.
- Exact SHA-256 and perceptual near-duplicate checks quarantine cross-partition content.
- Prepared catalogue: 30 independently QC-checked scenarios, 10 from each approved dataset, with
  1,529 verified slots.
- Scenario regression: baseline accuracy/balanced accuracy 88.358%/87.775%; enhanced model
  96.468%/96.270%. Baseline false-vacant/false-occupied/high-confidence-wrong counts were
  93/85/72; enhanced counts were 29/25/17.
- Prepared time lapses: three PKLot and three CNRPark+EXT sequences. No ACPDS video was fabricated
  because the available GoPro material does not form a defensible fixed-camera sequence.

## Video and media acceptance

- MP4 upload is supported; AVI is accepted when the local OpenCV/FFmpeg codec can decode it.
- Layout detection occurs on the reference frame and stable geometry is reused with registration
  checks; uncertain-motion frames receive null counts.
- Validation-only temporal selection chose EMA alpha 0.7, hysteresis 0.04, persistence 1, and no
  median window beyond the current frame.
- A prepared 20-frame PKLot run completed with zero dropped frames in 26.816 s, persisted 20
  timeline points and 127 events, and produced a playable 1,854,143-byte result.
- Progress polling, explicit cancellation, restart recovery, upload limits, invalid MP4 rejection,
  result persistence, cleanup policy, and safe media paths were exercised.
- A non-photographic synthetic image is rejected as `unsupported_layout` without fabricated counts
  or an analysis record. Advanced layout correction is optional, audited, and never required by the
  normal workflow.

## Product and accessibility acceptance

- `/` is a product landing page; the operational home is `/dashboard`.
- Thirteen Next.js routes build successfully, including image upload, video analysis, and advanced
  layout correction.
- The coherent product title is used throughout normal UI. Repeated offline-mode and release/handover
  branding was removed while internal engineering metadata was retained.
- Desktop and 390x844 mobile layouts were inspected. Mobile navigation is a keyboard-operable
  disclosure menu.
- Every button has a 44 px minimum target and visible focus treatment. History `Inspect`, Diagnostics
  `Inspect analysis`, and Reports `JSON + slots` use readable high-contrast states; reusable retry,
  presentation, scenario, overlay, and video controls were audited as well.
- Backend outage produces an explicit unavailable state and readable retry action. After restart,
  the same System page recovered to 9/9 readiness checks.

## Automated verification

- Ruff formatting/lint: passed.
- Backend pytest: 47/47 passed.
- Dataset archive validation: 7/7 passed.
- Integrated refinement verifier: passed.
- Frontend ESLint: passed.
- Frontend TypeScript check: passed.
- Next.js production build: passed, 13 routes.
- PowerShell script parsing: passed.
- Git whitespace check: passed.

## Hardware and runtime footprint

Verification used Windows, Python 3.12, an Intel Core Ultra 9 185H (16 cores/22 logical processors),
32 GB RAM, and an RTX 4070 Laptop GPU. Training used CUDA; deployment uses ONNX Runtime on CPU and
does not require a GPU. Approximately 14.5 MB of deployed ONNX vision weights plus small JSON
metadata are stored under the external data root. The full prepared/manifests/media footprint varies
with source extraction and uploaded videos; Git contains none of it.

## Honest limitations

Automatic localisation is validated for the twelve fixed-camera layouts represented by PKLot and
CNRPark+EXT. It is not moving-camera, dashcam, or universal unseen-camera support. Unfamiliar,
low-information, or unstable layouts are rejected or warned. Long-video throughput is CPU- and
slot-count-dependent; the measured 20-frame example processed at 0.746 analyzed frames/s. The
disclosed weak occupancy group and per-dataset differences remain future improvement targets.
