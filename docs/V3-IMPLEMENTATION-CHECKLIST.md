# V3 Implementation Checklist

Derived from the audit of commit `6989687`. Only **confirmed** defects appear here.

## Explicitly excluded (investigated and disproved during the audit)

| Suspected | Verdict |
|---|---|
| Video pane layout broken | **Not a defect.** Browser-pane letterboxing; measured 1280x720, `hOverflow: false`. |
| Duplicate GET requests = duplicate-job bug | **Not the same defect.** GETs double because of React 19 StrictMode in dev only. The duplicate *job* bug is separate and real (E-3). |
| Video metric desynchronisation | **Not a defect.** Verified correct at 11 seek positions; Vacant+Occupied = slot count everywhere. |
| Localizer corner accuracy poor | **Not a defect** as such — `mean_corner_error_normalized` 0.000181 is exact because geometry is *retrieved*, not regressed. The metric is misleading, not the geometry. |
| Model too small / needs replacing | **Not supported.** Corrected balanced accuracy is 99.49%. Keep MobileNetV3-Small unless evidence says otherwise. |

## P0 — correctness / misleading output

- [~] **P0-1** Unknown camera receives another camera's polygons (E-1). Generalized detector + OOD gate + verification state.
- [~] **P0-2** Degraded image matched to wrong lot at higher confidence than a correct match (E-2).
- [x] **P0-3** Missing PKLot `occupied` attribute silently becomes vacant (E-5). Quarantine + diagnostics + fail-safe.
- [x] **P0-4** `rectify_slot` processes full-resolution image per slot — 98% of ACPDS pipeline (L-1).

## P1 — reliability / performance

- [x] **P1-1** Models rebuilt on every request; ONNX sessions + SHA-256 per call (L-2).
- [x] **P1-2** `/model/status` builds five sessions per request (F).
- [x] **P1-3** ORB detector + reference descriptors recomputed every video frame (L-3).
- [x] **P1-4** Duplicate video jobs from concurrent POSTs (E-3). Frontend flag + backend dedupe.
- [x] **P1-5** Unlabelled predictions scored as "vacant" in diagnostics (E-4).
- [x] **P1-6** Temp `.working.mp4` leaks on failure; capture/writer not in `finally` (E-6).
- [x] **P1-7** Raw FFmpeg stderr surfaced to users (E-7).
- [x] **P1-8** FFmpeg discovery depends on a hardcoded home folder (E-8). Setting + readiness check.
- [x] **P1-9** Fresh video analyses labelled "Legacy analysis record" (D).

## P2 — UI / maintainability

- [x] **P2-1** "AI QUALITY" column presents confidence as correctness.
- [x] **P2-2** "Average inference time" mixes image latency with video wall time.
- [x] **P2-3** Upload/video runs listed as if they were datasets in analytics.
- [x] **P2-4** Scenario picker options indistinguishable.
- [x] **P2-5** Uploads identified by SHA-256 prefix instead of filename.
- [x] **P2-6** Video "Time" tile does not say which clock it shows.
- [x] **P2-7** "Insufficient localisation confidence" understates the real reason.
- [x] **P2-8** Advanced calibration opens in an unsaveable state.
- [ ] **P2-9** 10px horizontal overflow on `/analytics` at 375px.
- [x] **P2-10** Readiness responses expose absolute local paths.
      *Corrected during V3.5:* this was marked complete but `/health/ready`
      still returned the full dataset and media paths. Now redacted to the
      leaf name by `reliability_service.describe_location`.
- [ ] **P2-11** Extreme aspect ratios accepted (E-9).

## P3 — ML improvements

- [~] **P3-1** Generalized localization: OBB detector, leave-camera-out evaluation.
- [ ] **P3-2** Single-stage detect+classify vs two-stage benchmark.
- [ ] **P3-3** Occupancy: balanced sampling, augmentation, architecture comparison (validation only).
- [x] **P3-4** Three-state VACANT / OCCUPIED / UNCERTAIN with validation-selected bands.
- [x] **P3-5** Temporal hysteresis for video.
- [x] **P3-6** Verified-layout persistence + camera fingerprint reuse.

## New product work

- [x] **N-1** Landing page redesign (`/`).
- [x] **N-2** Layout verification / onboarding UX with AUTO DETECTED / VERIFICATION REQUIRED / USER VERIFIED states.
- [x] **N-3** Regression tests for every fixed defect.

## Protocol rules held throughout

- Protected holdout is **reporting only**. No architecture, threshold, augmentation, calibration or hyperparameter decision may touch it.
- Historical benchmark artifacts are preserved unmodified; corrected evaluations are written to new files.


## Status at last update

`[x]` complete and verified · `[~]` implemented, awaiting the trained detector · `[ ]` outstanding

**Verified by measurement**

| Item | Before | After |
|---|---|---|
| `rectify_slot`, ACPDS 125 slots | 9,135 ms | 142 ms (64x) |
| `GET /model/status` | 561 ms | 2.3 ms (244x) |
| `GET /health/ready` | 292 ms | 37 ms (7.9x) |
| `GET /dashboard/summary` | 322 ms | 11 ms (28x) |
| `GET /system` | 285 ms | 4.0 ms (71x) |
| ACPDS scenario analysis, mean | 3,056 ms | 478 ms (6.4x) |
| Camera-stability per frame | 47.9 ms | 15.2 ms (3.2x) |
| Concurrent duplicate video POSTs | 3 jobs | 1 job + two 409s |
| Occupancy false-vacant rate (validation) | 0.98% | 0.27% with the uncertain band |
| Scenario agreement (30 scenarios) | 97.4217% | 97.4213% (unchanged) |
| Backend tests | 47 | 69 |

**Outstanding**

- `P3-1` generalized detector: dataset and training pipeline built, leave-camera-out
  protocol verified, run in progress. Integration, ONNX export and the model
  comparison report follow once it finishes.
- `P3-2` single-stage detect-and-classify benchmark (dataset already built as
  `obb-occupancy`).
- `P3-3` occupancy architecture comparison on validation.
- Known-camera *uploads* and video currently ask for verification because the
  corroborating detector is not yet installed. This is the fail-safe behaving
  correctly, and it resolves when the detector lands. Prepared scenarios are
  unaffected and still analyse automatically.
