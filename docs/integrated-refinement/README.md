# Integrated refinement protocol

This refinement preserves frozen baseline commit
`2245350771a94088be41b9b51093686acbbbd06d` and adds six dependent capabilities in one controlled
delivery. The protected V2 artifacts are local runtime outputs and are deliberately absent from the
source archive.

## Change classification

| Area | Category | Evidence gate |
|---|---|---|
| ROI rectification, media safety, button contrast | BUG FIX | Unit tests and runtime checks |
| Enhanced occupancy classifier and expanded training protocol | MODEL IMPROVEMENT | Validation selection, then one protected holdout |
| Automatic slot localisation | NEW AI CAPABILITY | Per-dataset IoU recall/precision and confidence failure policy |
| Image upload, fixed-camera video, 30-scenario catalogue | NEW PRODUCT FEATURE | API and browser end-to-end checks |
| Landing page, title, navigation, wording, controls | UI/UX CHANGE | Lint, typecheck, build, keyboard/contrast audit |
| Hash quarantine, exposure disclosure, cleanup, recovery | INFRASTRUCTURE / VALIDATION | Integrity and reliability tests |

## Root-cause disposition

The frozen logistic baseline remains unchanged. Its application crop used a padded axis-aligned
bounding box, resized it to 32x32, and did not mask surrounding vehicles or rectify perspective.
Its training patches also did not perfectly reproduce full-image inference crops. These issues
combine with camera, site, rain, reflection, shadow, and dataset shifts. The scenario regression
report therefore separates false vacant, false occupied, high-confidence wrong, and threshold-
borderline errors. Human-reviewed QC overlays distinguish invalid/misaligned geometry from model
errors. Prepared scenarios are exposed regression evidence only and never tune architecture,
weights, calibration, or thresholds.

## Exact split policy

- PKLot occupancy: site/date groups; slot crops from one source image never cross partitions.
- PKLot localisation: fixed-camera site/date groups, with every site represented in train and
  disjoint dates assigned to validation and protected holdout.
- CNRPark+EXT occupancy: camera/day groups, respecting the available test metadata.
- CNR localisation: fixed-camera camera/day groups, with every camera represented in train and
  disjoint days assigned to validation and protected holdout.
- ACPDS: official Train, Validation, and Test groups.
- ACPDS localisation is excluded because its GoPro viewpoints move; its verified geometry remains
  supported for prepared occupancy analysis, but it is not evidence for fixed-camera auto-layout.
- Exact SHA-256 and perceptual-hash near duplicates crossing partitions are quarantined before
  training. Group and source-image leakage are rejected by verification.

The downloaded material was used during previous work; the new final holdout is consequently
described as protected from this refinement point forward, not historically virgin. A separately
licensed external confirmation set can be verified through the provided confirmation contract if
a historically independent claim is later required.

## Holdout discipline

1. Prepare and verify the immutable manifests.
2. Fit candidate models on train only.
3. Select architecture, threshold, and calibration on validation only.
4. Write checksummed decision locks.
5. Review development gates.
6. Open each final holdout once. Repeat evaluation is refused.

Do not regenerate manifests or retune after opening a final holdout. A failed final gate is reported
honestly and is not retried against the same protected data.

## Runtime boundaries

Automatic image localisation is primary. Images with too few/many detections or insufficient
localisation confidence return `unsupported_layout`; borderline detections return warnings. The
advanced correction UI is optional and every correction creates a new layout and audit row.

The deployed localizer recognizes the twelve fixed camera layouts present in PKLot and
CNRPark+EXT using a MobileNetV3-Small layout classifier, an observable aspect-family constraint,
and verified canonical polygons. It is not a universal parking-space detector. The experimental
one-stage polygon detector and ORB/RANSAC registration are retained in the development report as
rejected diagnostics. Uploaded images from unfamiliar cameras may be rejected and must never be
presented as validated automatic coverage.

Video support is limited to prerecorded fixed-camera MP4/AVI input. The reference layout is
detected once, checked with frame-to-reference homography, and reused only on stable frames. Motion-
uncertain frames have null counts rather than fabricated occupancy. Prepared videos are honest
same-camera/day time lapses, not claimed as native continuous capture.

## Acceptance summary

- 10+ confirmed scenarios per dataset, valid geometry, no exact/near duplicates
- 3 PKLot and 3 CNR prepared time lapses; zero fabricated ACPDS sequences
- Both enhanced ONNX models pass checksum and one-time holdout gates
- Automatic image success and explicit unsupported-layout behavior
- Video progress, cancellation, recovery, playback, timeline, and persistence
- Existing history, analytics, diagnostics, reports, presentation, retry and readiness behavior
- No raw/prepared datasets, runtime databases, media, weights, environments, dependencies, or
  secrets in Git
