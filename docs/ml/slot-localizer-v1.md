# Automatic parking-space localizer

The deployed primary image workflow is a fixed-camera layout recognizer. MobileNetV3-Small
classifies one of twelve registered PKLot/CNRPark+EXT camera layouts, an observable 16:9-versus-4:3
aspect-family constraint prevents cross-family confusion, and the system returns that layout's
verified canonical polygons. No user-supplied ROI coordinates are required in normal operation.

The experiment also trained a one-stage oriented-slot proposal network and evaluated ORB/RANSAC
template registration. Neither met the validation gate, so neither is presented as the deployed
primary method. Their metrics remain in the development report. ACPDS's moving GoPro viewpoints
are outside this fixed-camera contract; ACPDS remains supported for prepared occupancy analysis
using its verified supplied geometry.

Validation selects the layout policy and confidence threshold. The deployment gate evaluates IoU@0.50 and
IoU@0.75 recall, precision, corner error, count error, duplicates, missed slots, per-dataset
performance, worst layouts, CPU latency, and detected-geometry occupancy quality. Images outside
the validated domain return an unsupported-layout response rather than confident-looking counts.

This is a two-stage design: automatic fixed-camera layout recognition followed by the calibrated
occupancy classifier on rectified polygons. It exposes which stage failed, supports layout reuse
in fixed-camera video, and avoids relying on vehicle bounding boxes.

## Frozen result

- Train / validation / protected holdout images: 1,631 / 600 / 710
- Validation precision / recall at IoU 0.50: 94.1018% / 90.7222%
- Protected-holdout precision / recall at IoU 0.50: 93.5371% / 88.1501%
- Protected-holdout recall at IoU 0.75: 87.6933%
- PKLot precision / recall: 90.3320% / 81.7138%
- CNRPark+EXT precision / recall: 98.0355% / 98.1476%
- Detected-geometry occupancy accuracy: 97.3582% over 29,525 matched slots
- Measured CPU end-to-end image latency: mean 2,333.687 ms; p95 2,426.839 ms

The report retains complete failures caused by wrong camera-layout identification; therefore this
capability must be described as automatic analysis for registered fixed-camera views, not universal
arbitrary-camera localisation. Low-confidence or unfamiliar views return `unsupported_layout`.
