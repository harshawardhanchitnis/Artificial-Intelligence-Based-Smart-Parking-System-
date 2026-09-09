# V3.5 Implementation Checklist

Product-grade scene understanding. Measured findings are in
[V35-PRODUCT-ARCHITECTURE.md](V35-PRODUCT-ARCHITECTURE.md).

`[x]` complete and verified · `[~]` implemented, evidence partial · `[ ]` outstanding

## Preserved from V3 (verified not regressed)

- [x] Generalized unseen-camera parking-space detection
- [x] Known-layout corroboration rather than trusting a closed-set classifier
- [x] Invalid PKLot label quarantine
- [x] Optimised slot rectification
- [x] Model/session caching and fast `/model/status`
- [x] Optimised ORB reference reuse
- [x] Duplicate video-job prevention
- [x] H.264 browser-compatible video pipeline
- [x] Synchronised video metrics
- [x] Vacant / Occupied / Uncertain support
- [x] Improved analytics terminology
- [x] Redesigned product landing page
- [x] Schema and readiness integration (now v8, 13 checks)
- [x] Generalized detector tooling and evaluation scripts

All 25 V3 regression tests still pass.

## Part 1 — Benchmark Mode vs Product Mode

- [x] Benchmark mode uses exactly the dataset's slots, classifier only, no
      vehicle evidence, no fusion — historical figures stay comparable
- [x] Benchmark responses carry `analysis_mode: "benchmark"`
- [x] Product mode analyses the whole visible scene
- [x] Product mode never claims ground-truth agreement for unlabelled uploads
- [x] `ml/compare_modes.py` reports both for the same scenes
- [x] Contract locked by `backend/tests/test_v35_modes.py`

## Part 2–3 — Geometry representation and architecture selection

- [x] Rotated-rectangle ceiling measured: 0.907 mean IoU unseen, 0.762 on ACPDS
- [x] Four-corner (`pose`), segmentation (`seg`) and oriented-box (`obb`)
      datasets built from one manifest on identical camera-aware splits
- [x] Four-corner model trained and evaluated
- [x] Oriented-box control trained on identical data — isolates representation
      from data
- [~] Segmentation candidate — first run died on host memory, second thrashed
      VRAM at 11 min/epoch; re-queued at batch 4
- [~] Four-corner over all sources — fills the representation × data cell
- [x] RT-DETR eliminated on capability (axis-aligned output only) and
      benchmarked where axis-aligned output is correct, as a vehicle detector
- [x] Selection on `val_unseen` only; `test_unseen` never opened

## Part 4 — Geometric / line refinement

- [x] Implemented with support clustering, shift caps and a brightness gate
- [x] Measured across three tightenings: best result was neutral
- [x] Root cause established — PKLot bays are annotated around vehicles, not on
      the paint (edge-response ratio 0.91 against ACPDS's 2.27)
- [x] Shown to be inseparable at inference time
- [x] Shipped **disabled** behind `settings.enable_marking_refinement`

## Part 5 — Full-scene vehicle detection

- [x] Five candidates benchmarked on unseen cameras
- [x] Resolution swept: 1280 selected at the knee (0.931 recall, 191 ms)
- [x] Confidence swept and the trade-off recorded
- [x] Runs on the whole frame, independent of bay geometry
- [x] ONNX export + runtime + readiness check

## Part 6 — Slot ↔ vehicle association

- [x] Asymmetric coverage (intersection-over-minimum), not IoU
- [x] Perspective-aware ground footprint
- [x] Lorry across two bays, two-wheeler in one bay, aisle vehicle, corner
      clipping, two two-wheelers sharing a bay — all covered by tests

## Part 7 — Occupancy fusion

- [x] Log-odds model; no signal can veto another
- [x] Coefficients fitted on the occupancy validation split
- [x] Uncertain band selected by sweeping the error target to the knee
- [x] Confirmed out-of-fold: false-vacant 0.61% → 0.49%
- [x] Degrades to classifier-only when no artifact or detector is present

## Part 8 — Three occupancy states

- [x] Vacant / Occupied / Uncertain throughout image and video paths
- [x] Thresholds selected on validation only
- [x] Video counts partition the layout exactly (regression-tested)

## Part 9–10 — Automatic calibration, no human in the loop

- [x] Multi-frame consensus calibration for fixed cameras
- [x] Frames sampled spread across the clip, not consecutively
- [x] `AUTOMATIC_LAYOUT_UNRESOLVED` — the AI's own verdict, asks for nothing
- [x] Vehicles still reported when the layout is unresolved
- [x] Layout lifecycle persisted (`calibrating` / `active` / `recalibrating`)
- [x] Drift detection against independently detected geometry
- [x] Correction endpoint slot cap raised 300 → 400 to match the detector, which
      was the source of the bare "Request validation failed"
- [x] Manual correction demoted to an explicit advanced action

## Part 11 — Unmapped vehicles

- [x] Reported as a neutral count in API, overlay and interface

## Part 12 / 19 — Product visualisation and copy

- [x] Green / red / amber bay polygons in image and video overlays
- [x] Vehicles drawn thinner and cooler so the parking map stays dominant
- [x] Unmapped vehicles drawn distinctly
- [x] Only Car / Two-wheeler / Truck ever shown
- [x] Deployment-dependent privacy copy (`frontend/src/lib/deployment.ts`)
- [x] Landing page no longer describes manual correction as the workflow
- [x] History detail no longer claims correctness for unlabelled uploads

## Part 14 — Dataset and training review

- [x] CNRPark+EXT annotations shown to be axis-aligned (46% of the corpus)
- [x] PKLot viewpoint diversity shown to be one training site
- [x] Occupancy balance shown to be ~50/50, ruling out class imbalance
- [x] Vacant-bay blindness quantified (0.433 against 0.711)

## Part 15 — Vehicle class mapping (corrected)

- [x] Three classes only; source taxonomy never exposed
- [x] **`bus` is NOT mapped to `TRUCK`**; `bicycle` is not mapped to
      `TWO_WHEELER`. Both are unsupported, asserted by test
- [x] The size-based class conversion that briefly existed is removed
- [x] Unsupported but vehicle-shaped objects are kept as *unclassified*: they
      carry occupancy evidence and never a class
- [x] Unit-tested that no mapping can introduce a fourth class
- [~] Two-wheeler validation on a separate general-domain development set
      (COCO val2017, outside the parking protocol)

## Parking-area understanding (new)

- [x] Facility extent inferred from bay geometry plus parked-vehicle footprints
- [x] Non-rectangular, morphological closing scaled to local bay size
- [x] Three categories: in mapped space / in parking area / outside
- [x] Operational figures exclude off-site vehicles
- [x] `area_unknown` when no layout exists, rather than asserting position
- [x] CNRPark camera1: 40 "unmapped" became 37 in-area + 3 outside

## Vehicle false positives (new)

- [x] Precision and false-detections-per-image measured at the operating point
- [x] Single-image size/aspect/confidence filters measured and **rejected** —
      false detections are larger and barely less confident than true ones
- [x] Temporal confirmation for video (2 of the last 3 frames)
- [x] Parking-area context excludes off-site detections
- [ ] Re-measure false detections per image on video with confirmation active

## Non-YOLO geometry candidate (new)

- [~] Keypoint R-CNN (ResNet50-FPN, two-stage) trained on the same dataset and
      scored by the same evaluator through an adapter
- [x] VRAM and runtime recorded so infeasibility would be evidenced, not assumed

## Part 17 — Performance

- [x] Vehicle detector CPU latency measured across five candidates and four
      resolutions
- [x] Space model CPU latency measured per candidate
- [ ] Per-stage product pipeline profile — first attempt was contaminated by
      concurrent GPU training; must be re-run on an idle machine

## Part 18 — Video product pipeline

- [x] Automatic multi-frame calibration for an unknown camera
- [x] Geometry cached; space detection is not re-run per frame
- [x] Full-scene vehicle detection and fusion per sampled frame
- [x] Fuse-then-smooth ordering, so vehicle evidence is not counted repeatedly
- [x] Validated against catalogue ground truth: mean error 3.75 spaces overall,
      **0.78 spaces** on frames the system reported as settled

## Part 20 — Testing

- [x] 141 backend tests pass (25 V3 + 69 new + existing)
- [x] Frontend lint, typecheck and production build pass
- [x] All 17 backend GET endpoints return 200
- [x] All 13 frontend routes render
- [x] Video job end-to-end, including H.264 playback

## Unseen-camera sequences (new)

- [x] `ml/build_development_sequence.py` reconstructs a chronological
      fixed-camera time-lapse from `val_unseen` stills (never the holdout)
- [x] Two built: PKLot UFPR04 (24 frames), CNRPark camera7 (14 frames)
- [x] Calibration measured on both: camera7 recovers 90.2% of spaces between
      disjoint halves at IoU 0.5, median polygon IoU 0.941
- [x] **Stated precisely:** both cameras are *registered identities* for the
      recall path, so a video job resolves before calibration is consulted. The
      calibration branch is therefore driven directly in
      `tests/test_v35_calibration.py` with the frames a job would have used.

## Model selection (complete)

- [x] Four candidates trained and evaluated by one evaluator on one sample,
      each scored the way it would actually run
- [x] **Selected: YOLO11n-pose, four independent corners**, installed as
      `parking-space-detector-v3` with the previous artifact preserved
- [x] Keypoint R-CNN measured as the geometry upper bound and *not* selected on
      deployment cost (5,697 ms CPU, 236 MB) -- recorded with its numbers
- [x] Segmentation eliminated on measurement (corner error 0.830)
- [x] Corroboration safeguard re-measured after the swap: improved, 9/9
- [x] Drift threshold re-measured after the swap: was wrong, now 0.15
- [x] Known-camera behaviour verified unregressed (exact slot counts, 1.000
      agreement on every registered camera)
- [x] `test_unseen` opened once, reporting only

## Verification (complete)

- [x] 173 backend tests pass
- [x] ruff clean across `backend` and `ml`
- [x] Frontend lint, typecheck and production build pass
- [x] All 17 backend GET endpoints return 200; readiness 13/13
- [x] All 13 frontend routes render
- [x] Video job end to end on an unseen-camera sequence, with facility-scoped
      vehicle counts and H.264 playback
- [x] P2-10 corrected: readiness had still been publishing absolute paths
- [x] Serving refactor verified: 190 backend tests pass (was 173), ruff clean,
      frontend lint + typecheck + production build pass, 31/31 GET routes
      answer, readiness 13/13, video job end to end with duplicate protection
      and H.264 playback
- [x] **Torch removed from the serving path** -- three modules split into
      serving and training halves; `import app.main` now loads no torch,
      torchvision or ultralytics (0 modules, was 1,233), and the boundary is
      enforced by `tests/test_v35_serving_runtime.py`
- [x] **The real serving defect found and fixed** -- the 10x figure attributed
      to torch did not survive an interleaved A/B test (ratio 1.02). The cost
      was ONNX Runtime sizing four concurrent sessions' thread pools to all 22
      logical processors; capping them at 3 took the pipeline from 9,478 ms to
      1,499 ms and a product request from 11,396 ms to 2,517 ms
- [x] Presentation Mode parity pinned: the showcase serves scenario identifiers
      only, and its numbers come from the same `POST /analysis/scenarios/{id}`
      the Analyse page uses. A manual comparison of PUCPR *cloudy* against PUCPR
      *sunny* read as a data bug; the showcase now prints the scenario id and
      the verified label counts so the comparison is auditable on screen
- [x] Output equivalence proven: torch removal is byte-identical across 30
      scenarios in both modes; the thread cap changes no verdict, class or
      count across 120 images, 3,925 bays and 2,676 vehicles

## Outstanding

- [ ] Two-wheeler validation in the parking domain (no such data exists)
- [ ] An end-to-end video job that onboards a camera unseen to *every* path
- [ ] Viewpoint diversity: one PKLot training site is the binding constraint on
      unseen-camera recall, not the architecture
