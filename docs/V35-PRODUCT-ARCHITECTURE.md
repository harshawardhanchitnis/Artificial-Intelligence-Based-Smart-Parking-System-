# V3.5 — product-grade scene understanding

Measured findings and architecture decisions for the move from "classify the
bays somebody gave us" to "understand the parking scene".

Every number here was measured on this machine against the prepared corpus. The
protected occupancy holdout and the localization `test_unseen` split were not
opened for any selection decision.

---

## 1. What was actually wrong

The V3 detector reported 72.97% mAP50 on unseen cameras. That number is real and
it is also close to useless as a description of the product, for four reasons
that were each measured separately.

### 1.1 A rotated rectangle cannot represent a parking bay

The training labels store the true four corners of each bay. An oriented-box
head cannot emit them — it regresses centre, width, height and angle — so
Ultralytics fits a minimum-area rotated rectangle to the corners before
training. That fit is a ceiling no amount of training can lift.

`ml/measure_representation_ceiling.py`, measured over every annotated bay:

| Split | Bays | Mean ceiling IoU | Below 0.90 | Below 0.75 |
|---|---:|---:|---:|---:|
| val_unseen | 21,271 | 0.9066 | 47.4% | 5.4% |
| val_known | 10,455 | 0.9297 | 33.8% | 1.1% |

Broken down by camera family, the loss is entirely concentrated where
perspective is strong:

| Camera family | Bays | Mean ceiling IoU | Below 0.75 |
|---|---:|---:|---:|
| ACPDS (GoPro, wide angle) | 1,685 | **0.7615** | **46.4%** |
| PKLot UFPR04 | 10,248 | 0.8453 | 3.6% |
| PKLot PUCPR | 5,800 | 0.8734 | 2.0% |
| CNRPark+EXT | 13,993 | **1.0000** | 0.0% |

A perfect oriented-box model would still fail an IoU-75 match on nearly half of
all wide-angle bays.

### 1.2 Forty-six per cent of the geometry labels contain no geometry

CNRPark+EXT's ceiling IoU is exactly 1.0000 because **every one of its 62,668
annotations is an axis-aligned rectangle** — verified directly: 45,702 of 45,702
sampled instances have only two distinct x values and two distinct y values.
They are upright boxes around each slot, not bay outlines.

Those 62,668 instances are 46.1% of the 135,802-instance corpus, and 48.3% of
the training split. Nearly half the geometry training signal was teaching the
model that a parking space is an upright rectangle.

### 1.3 The aggregate hides a near-total failure

Scored with one matcher on the geometry-faithful unseen split (PKLot + ACPDS),
the shipped detector finds **19.5% of bays at IoU 0.5 and 1.8% at IoU 0.75**:

| Group | Bays | recall@50 | recall@75 | precision@50 | matched IoU |
|---|---:|---:|---:|---:|---:|
| ALL | 11,933 | 0.195 | 0.018 | 0.288 | 0.454 |
| ACPDS | 1,685 | 0.479 | 0.087 | 0.497 | 0.557 |
| PKLot | 10,248 | 0.148 | 0.006 | 0.235 | 0.423 |

The 72.97% aggregate is carried by CNRPark+EXT, whose axis-aligned annotations
are the easiest possible target and which contributes 9,338 of the 21,271
val_unseen bays.

### 1.4 The detector learned "car", not "parking space"

Per-bay recall split by the bay's own occupancy label:

| Slot state | Bays | recall@50 | recall@75 |
|---|---:|---:|---:|
| Vacant | 2,157 | **0.433** | 0.182 |
| Occupied | 2,488 | **0.711** | 0.287 |

On PKLot the gap is 13.8×: 2.4% of vacant bays found against 33% of occupied
ones. This is not class imbalance — the training split is 50.3% occupied. It is
salience: a parked vehicle is a far stronger visual signal than a painted line.

For a product that exists to report *free* spaces, this is the worst possible
failure direction, and it is invisible in any aggregate metric.

### 1.5 The training corpus has almost no viewpoint diversity

| Split | PKLot sites |
|---|---|
| train | **PUCPR only** |
| val_known | PUCPR |
| val_unseen | **UFPR04 only** |
| test_unseen | UFPR05 only |

Unseen-camera generalisation for PKLot is measured on exactly one camera, and
the model has seen exactly one PKLot viewpoint in training. Total training
viewpoint diversity across the corpus is one PKLot site, nine CNRPark cameras,
and 227 single-image ACPDS captures.

---

## 2. Architecture decisions

### 2.1 Parking-space geometry: four independent corners

**Options considered.** Four-corner keypoint regression (YOLO pose); instance
segmentation with contour reconstruction; transformer detection (RT-DETR);
oriented box (incumbent); hybrid neural proposal plus classical line
refinement.

RT-DETR was eliminated on capability rather than on a benchmark: in Ultralytics
it emits axis-aligned boxes only, so it cannot express a quadrilateral at all
and would sit *below* the incumbent's 0.907 representation ceiling, not above
it. It was still benchmarked as a vehicle detector, where axis-aligned output is
the correct representation — see §2.3.

**Datasets built** (`backend/app/datasets/quad_dataset.py`), all from one
manifest on identical camera-aware splits so the comparison is not confounded:

- `quad-pose-geom` — box plus four visible keypoints, PKLot + ACPDS
- `quad-seg-geom` — the quadrilateral as an instance mask, same sources
- `quad-obb-geom` — the incumbent representation, same sources (the control
  that separates "changed the representation" from "changed the data")

**Measured** (`ml/evaluate_space_geometry.py`, val_unseen = PKLot + ACPDS, one
matcher and one set of rules for every candidate).

Two perspectives are reported because the corpus supports two different
questions. **Dataset geometry** (the `ALL` column) asks how closely predictions
match the annotations that exist. **Product geometry** (the `ACPDS` column) asks
how closely they follow the real bay boundaries — and only ACPDS can answer it,
because its bays are drawn on the painted markings while PKLot's are drawn
around where vehicles stand (§2.2).

Dataset geometry:

| Candidate | Representation | Training data | CPU ms | rec@50 | rec@75 | prec@50 | polygon IoU | corner err |
|---|---|---|---:|---:|---:|---:|---:|---:|
| obb-incumbent | rotated box | all sources | 842 | **0.195** | 0.018 | **0.288** | 0.454 | 0.479 |
| obb-geom (control) | rotated box | geometry-faithful | 228 | **0.215** | 0.008 | 0.131 | 0.415 | 0.546 |
| pose-geom | four corners | geometry-faithful | 378 | 0.151 | **0.025** | 0.279 | **0.455** | **0.443** |
| pose-all | four corners | all sources | **193** | 0.115 | 0.013 | 0.176 | 0.356 | 0.579 |

Product geometry (ACPDS — where the boundary is knowable):

| Candidate | rec@50 | rec@75 | prec@50 | polygon IoU | corner err |
|---|---:|---:|---:|---:|---:|
| obb-incumbent | **0.479** | 0.087 | 0.497 | 0.557 | 0.450 |
| obb-geom (control) | 0.189 | 0.033 | 0.246 | 0.432 | 0.609 |
| **pose-geom** | 0.449 | **0.123** | **0.598** | **0.621** | **0.291** |
| pose-all | 0.431 | 0.081 | 0.542 | 0.574 | 0.354 |

#### The representation comparison, isolated

`obb-geom` and `pose-geom` were trained on identical data with an identical
schedule, so the difference between them is the representation and nothing else.
On product geometry the four-corner head gives **+138% recall, +143% precision,
+44% polygon IoU and −52% corner error**. That is the cleanest evidence in this
work, and it is the reason the representation changed.

#### Measured against the deployed artifact, not only the checkpoint

Every row above scores a training checkpoint through Ultralytics. That is not
what ships. Ultralytics letterboxes a `.pt` into a *rectangle* — a 1280x720
frame becomes 1024x576 — while the exported ONNX graph has a fixed square input,
so the deployed model sees the same frame padded to 1024x1024, and the runtime
suppresses on polygon overlap rather than on axis-aligned boxes.

The difference is real: on one frame's top detection the two agree at 0.915
polygon IoU, not 1.0. So the evaluator gained an `onnx` mode that scores the
artifact the application actually loads, and the deployed path measures
*better* than the checkpoint it came from:

| pose-geom, scored as | ACPDS rec@50 | prec@50 | polygon IoU | corner err |
|---|---:|---:|---:|---:|
| PyTorch checkpoint (398 images) | 0.449 | 0.598 | 0.621 | 0.291 |
| **deployed ONNX (150 images)** | **0.507** | **0.678** | **0.638** | **0.260** |

Scored that way, on an identical 150-image sample, the two shipped-path
candidates trade off rather than one dominating:

| Deployed ONNX | ALL rec@50 | ALL rec@75 | ACPDS rec@50 | ACPDS rec@75 | ACPDS prec@50 | ACPDS IoU | ACPDS corner err | vacant rec@50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| obb-incumbent | **0.216** | 0.021 | **0.566** | 0.103 | 0.604 | 0.588 | 0.373 | 0.088 |
| pose-geom | 0.164 | **0.029** | 0.507 | **0.144** | **0.678** | **0.638** | **0.260** | **0.093** |

The oriented box **finds more bays**; the four-corner head **puts them in the
right place** — 40% better recall at the strict IoU 0.75, 12% better precision,
30% lower corner error. Its recall deficit is largely a data deficit: it was
trained on 490 images against the incumbent's 1,488, and the controlled pair
(§ above) shows that on identical data the four-corner head wins on recall too,
by 138%.

The gain comes from the runtime's polygon NMS, which suits densely packed
slanted bays better than the axis-aligned box NMS Ultralytics applies. This also
means the shipped OBB detector's row understates or overstates it by an unknown
amount for the same reason, and any figure quoted as *product* performance must
come from the `onnx` mode.

#### Swapping the detector must not weaken the V3 safety fix

The V3 work fixed a P0 defect: an unfamiliar car park was answered with a
registered camera's polygons. The fix trusts a recalled layout only when it
explains at least 35% of what the detector independently finds, so changing the
detector changes that safeguard's margins and it has to be re-measured, not
assumed.

Nine unseen-camera frames, each scored against its own layout and against eight
layouts belonging to other car parks:

| Detector | Correct layout (median / min) | Wrong layout (median / max) | Correct layouts passing | Wrong layouts wrongly passing |
|---|---|---|---:|---:|
| obb-incumbent | 0.500 / 0.321 | 0.060 / 0.143 | 8 / 9 | 0 / 40 |
| **pose-geom** | **0.643 / 0.400** | 0.102 / 0.214 | **9 / 9** | 0 / 40 |

The four-corner detector *strengthens* the safeguard. Correct layouts score
higher, all nine clear the threshold where the incumbent missed one, and the gap
between the worst correct score (0.400) and the best wrong score (0.214) is
wider than the incumbent's (0.321 against 0.143 — a narrower absolute margin at
a lower level). No wrong layout passes under either detector.

#### The data ablation cuts opposite ways for the two representations

| Representation | geometry-faithful only | + CNRPark (axis-aligned) | Effect on ACPDS polygon IoU |
|---|---:|---:|---|
| rotated box | 0.432 | 0.557 | **helped** (+0.125) |
| four corners | 0.621 | 0.574 | **hurt** (−0.047) |

This was not the expected result and it is worth stating precisely. CNRPark's
62,668 axis-aligned annotations are usable *volume* for a model that predicts
rectangles — it triples the training set and the labels are not wrong for that
head. For a model that predicts four independent corners they are actively
misleading: every one of them says "this bay is an upright rectangle", and the
corner regressor learns it. Corner error rises from 0.291 to 0.354 and polygon
IoU falls when they are added.

So the earlier hypothesis in §1.2 — that CNRPark's labels are simply harmful —
was half right. They are harmful *to a geometry model* and helpful to a
rectangle model, and the shipped OBB detector's apparent advantage on ACPDS
recall comes from that extra volume rather than from its representation.

### 2.1b The full candidate field

Four architectures, one evaluator, one 150-image sample, each scored the way it
would actually run.

| Candidate | Family | Representation | ALL rec@75 | ACPDS rec@50 | ACPDS rec@75 | ACPDS prec@50 | ACPDS IoU | corner err | vacant rec@50 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| obb-incumbent | YOLO one-stage | rotated box | 0.021 | **0.566** | 0.103 | 0.604 | 0.588 | 0.373 | 0.088 |
| pose-geom | YOLO one-stage | four corners | 0.029 | 0.507 | 0.144 | **0.678** | 0.638 | 0.260 | 0.093 |
| seg-geom | YOLO one-stage | mask -> contour | 0.014 | 0.273 | 0.085 | 0.213 | 0.452 | 0.830 | 0.048 |
| keypoint-rcnn | **two-stage R-CNN** | four corners | **0.047** | 0.542 | **0.340** | 0.591 | **0.667** | **0.203** | **0.095** |

**Instance segmentation is decisively out.** Its corner error is 0.830 -- four
times the best -- and its ACPDS precision is 0.213. Recovering a quadrilateral
from a predicted mask contour loses more geometry than the mask gains: the
contour is ragged, `approxPolyDP` frequently will not reduce it to four points,
and the minimum-area-rectangle fallback reintroduces exactly the rectangle
assumption the four-corner work exists to remove. It was also the only candidate
that could not be trained at the common settings (see §2.1c).

**The non-YOLO candidate leads on geometry.** Keypoint R-CNN reaches 0.340
recall at the strict IoU 0.75 on ACPDS, 3.3x the incumbent and 2.4x the YOLO
four-corner head, with the lowest corner error and the highest polygon IoU. Two
confounds are measured rather than assumed, in §2.1c.

### 2.1c Feasibility and confounds, measured

Nothing here was skipped for being awkward; where a candidate needed different
settings, the deviation is recorded.

| Candidate | Peak VRAM | Train time | Settings deviation |
|---|---:|---:|---|
| obb-geom | ~6 GB | 60 epochs | none |
| pose-geom | ~6 GB | 5,120 s / 60 epochs | none |
| pose-all | ~6 GB | 3,783 s / 60 epochs | none |
| seg-geom | **7.44 GB at 1024/batch 4** | — | **failed**: 18.2 s/iteration thrashing an 8 GB card, ~30 h projected |
| seg-geom (shipped run) | 3.83 GB at 768/batch 2 | 1,494 s / 12 epochs | **imgsz 768, batch 2, mosaic off** |
| keypoint-rcnn | 3.73 GB | 1,721 s / 10 epochs | 10 epochs, batch 2 |

Segmentation additionally required mosaic to be disabled: mosaic stitches four
images into one, and with ~76 bays per image the instance count reached 266,
which made the mask allocator ask for a `(266, 256, 256)` array and exhaust host
memory. Three earlier failures across candidates were all host pagefile
exhaustion from dataloader shared memory, not VRAM -- every model fits the card
comfortably once nothing else competes.

### 2.2 Marking-based geometry refinement: implemented, measured, disabled

Investigated per the brief and **not adopted**, on evidence.

Perturbing annotated bays by 0.10 of their own scale and asking whether
refinement pulled them back:

| Configuration | Accepted | Mean IoU before → after | Improved / worsened |
|---|---:|---|---|
| Initial | 86.9% | 0.7522 → 0.7294 | 150 / 302 |
| + offset clustering | 63.1% | 0.7522 → 0.7481 | 153 / 175 |
| + brightness gate | 3.7% | 0.7522 → 0.7510 | 6 / 13 |

The cause is a property of the corpus. Sampling marking response along
annotated bay edges against ground displaced away from them:

| Dataset | On GT edge | Displaced | Ratio |
|---|---:|---:|---:|
| ACPDS | 0.1248 | 0.0549 | **2.27×** |
| PKLot | 0.0464 | 0.0511 | **0.91×** |

ACPDS bays are drawn on the paint. PKLot bays are drawn around where vehicles
stand, so on PKLot the markings and the ground truth are different things and
moving toward the paint provably moves away from the truth. PKLot is 86% of the
evaluation bays.

Nor can the two be separated at inference time, which is what would have
rescued the feature: PKLot car parks *are* painted, so scene marking response
does not distinguish them (ACPDS 0.293, PKLot 0.263, overlapping ranges). The
difference is in the annotation convention, which is not visible in the image.

The machinery is kept, guarded and tested behind
`settings.enable_marking_refinement`, defaulting off, and should be reconsidered
against a corpus whose bay annotations follow the markings.

### 2.3 Vehicle understanding: YOLO11n at 1280

Five candidates, scored on unseen cameras by whether they find a vehicle in a
bay the dataset labels occupied, and whether they invent one in a bay it labels
vacant (`ml/evaluate_vehicle_detector.py`, CPU, imgsz 960):

| Candidate | Architecture | CPU ms | Occupied-bay recall | Vacant-bay false rate | Size MB |
|---|---|---:|---:|---:|---:|
| **yolo11n** | CNN detect | **136** | **0.881** | 0.039 | **5.6** |
| yolo11s | CNN detect | 240 | 0.859 | 0.038 | 19.3 |
| yolo11s-seg | CNN instance-seg | 353 | 0.843 | 0.039 | 20.7 |
| rtdetr-l | Transformer (DETR) | 1102 | 0.827 | 0.043 | 66.5 |
| yolo26n | CNN detect | 146 | 0.774 | 0.039 | 5.5 |

The smallest model won on both accuracy and latency. The transformer was 8×
slower, 12× larger and worse. Instance segmentation cost 2.6× the latency for
lower recall; its masks would have improved association, but association was not
the binding constraint.

Input resolution mattered far more than architecture:

| imgsz | CPU ms | Occupied-bay recall | Vacant-bay false rate |
|---:|---:|---:|---:|
| 640 | 100 | 0.712 | 0.037 |
| 960 | 134 | 0.881 | 0.039 |
| **1280** | **191** | **0.931** | 0.045 |
| 1600 | 258 | 0.956 | 0.046 |

**Selected: YOLO11n at 1280** — the knee of the curve. 1600 buys 2.5 more points
of recall for 35% more latency.

For context, the *space* detector finds 36% of occupied bays. The vehicle
detector finds 93%. Full-scene vehicle detection is the capability that was
missing.

### 2.4 Three-class vehicle taxonomy

`backend/app/ml/vehicle_taxonomy.py`. The taxonomy is a **closed set of three**,
and membership is decided by what a source label means, never by how large the
object looks.

| Product class | COCO sources | Other source labels |
|---|---|---|
| `CAR` | `car` | suv, van, minivan, hatchback, sedan |
| `TWO_WHEELER` | `motorcycle` | scooter, motorbike, moped |
| `TRUCK` | `truck` | lorry, pickup |

**`bus` is not mapped to `TRUCK`.** Nor is `bicycle` mapped to `TWO_WHEELER`.
Both are unsupported, and the exclusion is asserted by
`test_bus_is_never_mapped_to_truck_by_any_route`.

An earlier iteration did fold `bus` into `TRUCK` on the reasoning that a parked
coach should not vanish. That was wrong on the taxonomy and wrong in practice,
and the measurement is worth recording: on a steep top-down PKLot view the
detector labels ordinary cars `bus`, and one frame surfaced **31 trucks in a car
park that contained none**. A scene-relative size correction was then tried and
also rejected — deciding a vehicle's class from its size is not classification.

#### Unsupported objects are evidence, not a fourth class

Excluding a label does not delete the object. On that same frame the model
scores those cars `bus` 0.33 and `car` 0.007, so no rule over its own
probabilities recovers them as cars — the detector is simply wrong there, and
confidently so. Ranking within the supported classes was measured to delete 139
real objects from the frame.

Anything vehicle-shaped is therefore kept, and only its *class* is withheld:

```
detected  ->  maps to a product class?  ->  yes: CAR / TWO_WHEELER / TRUCK
                                        ->  no : unclassified object
```

An unclassified object contributes occupancy evidence exactly like a classified
vehicle — it fills the bay it stands in — and is excluded only from the class
counts. So a coach can never appear as a lorry, a misclassified car can never
appear as anything at all, and neither disappears from the occupancy decision.

Measured effect of the correction, same frames:

| Scene | Before (bus→TRUCK) | After |
|---|---|---|
| PKLot PUCPR, rainy | 3 CAR, **31 TRUCK** | 3 CAR, 0 TRUCK, 31 unclassified |
| ACPDS gopr6543 (2 real coaches) | 13 CAR, **2 TRUCK** | 13 CAR, 0 TRUCK, 2 unclassified |
| ACPDS gopr6712 (3 real lorries) | 42 CAR, 3 TRUCK, 1 bus→TRUCK | 42 CAR, **3 TRUCK**, 1 unclassified |

Real lorries keep their class; coaches and misclassifications do not acquire one.

**Limitation, unchanged and unresolved by this correction:** the parking corpus
holds roughly one two-wheeler per thousand detections, so `TWO_WHEELER`
accuracy cannot be measured on it. See §2.8.

### 2.5 Parking-area understanding

`backend/app/ml/parking_area.py`. Detecting vehicles across the whole frame
finds cars the bay map missed — and also traffic that has nothing to do with the
site. One CNRPark view reported **40 unmapped vehicles**, most of them on the
road behind the car park, which would have inflated every operational figure.

The facility's extent is derived, never configured. Bay polygons *and* the
ground footprints of detected vehicles are painted into a mask and
morphologically closed at a radius scaled to the local bay size. Aisles close,
because they have bays on both sides; a road does not, because it has none on
its far side.

Vehicles seed the mask as well as bays because a benchmark rarely labels the
whole site: CNRPark camera 1 annotates two rows of a five-row car park, and a
region grown from bays alone put forty genuinely parked cars outside the
facility.

Three categories, of which only the first two describe the site:

| Category | Meaning |
|---|---|
| `in_mapped_space` | associated with a detected bay |
| `in_parking_area` | on the site, in no detected bay — aisle, or an undetected bay |
| `outside_parking_area` | off-site: excluded from every operational figure |
| `area_unknown` | no layout, so no area: position is not asserted |

Measured on the case that prompted it:

| Scene | Mapped | In area | Outside | (before) |
|---|---:|---:|---:|---|
| CNRPark camera1 | 33 | **37** | **3** | 33 mapped, 40 "unmapped" |
| ACPDS gopr6543 | 4 | 4 | 7 | — |
| PKLot PUCPR rainy | 27 | 1 | 6 | — |

The known risk is stated rather than hidden: because parked vehicles seed the
region, a dense stationary queue on an adjacent road could merge into it. A
kerb-width gap larger than the closing radius is what prevents it, and that gap
is a property of the site rather than a guarantee.

### 2.6 Vehicle false positives

Measured at the shipped operating point (YOLO11n, 1280, conf 0.25) on unseen
cameras, adjudicated by whether a detection occupies a bay the dataset labels
occupied or vacant:

| Group | Occupied-bay recall | Precision in labelled bays | False detections per image |
|---|---:|---:|---:|
| ALL | 0.924 | **0.9847** | **0.267** |
| ACPDS | 0.887 | 0.9808 | 0.350 |
| CNRPark+EXT | 0.948 | 0.9888 | 0.250 |
| PKLot | 0.940 | 0.9833 | 0.200 |

**No single-image suppression rule is applied, because none is supported by the
data.** Detections landing in labelled-vacant bays were compared against those
in occupied bays: they are *larger*, not smaller (relative footprint 1.70
against 1.01), and barely less confident (0.63 against 0.71). A size, aspect or
confidence filter would have removed true vehicles at the same rate as false
ones. Raising confidence globally was measured and rejected for the same reason:
0.25 → 0.45 cuts the vacant-bay false rate from 4.5% to 2.6% but costs 12 points
of occupied-bay recall.

What does work is evidence a still image does not have:

- **Parking-area context** (§2.5) removes off-site detections from the figures.
- **Temporal confirmation** (`backend/app/ml/vehicle_tracking.py`) requires a
  detection to recur in two of the last three sampled frames before a video
  counts it. A parked vehicle is in the same place every frame; a numeral, a
  drain cover or a shadow is not.
- **Fusion** weighs vehicle evidence against the crop classifier rather than
  letting either decide alone, so a persistent false detection on empty tarmac
  is contradicted by a classifier that sees empty tarmac.

### 2.7 Slot ↔ vehicle association

`backend/app/ml/association.py`. Intersection-over-union gets the two commonest
cases backwards: a lorry straddling two bays overlaps each at ~0.5 IoU, and a
scooter in a full-size bay overlaps at ~0.15, yet both bays are occupied.

The two coverage directions are kept separate and a bay is occupied when
*either* is high — intersection-over-minimum. A vehicle's box is also cropped to
its lower 45% before comparison, because the box includes the vehicle's height,
which projects away from the camera and would otherwise drag associations into
the next bay.

This was caught by a regression test, not by inspection: the first threshold
pair left a lorry across two bays occupying neither.

### 2.8 Occupancy evidence fusion

`backend/app/ml/occupancy_fusion.py`, coefficients fitted by
`ml/fit_occupancy_fusion.py` on the occupancy **validation** split (260 images,
13,860 bays, 5,490 with vehicle evidence). Log-odds addition, not a rule ladder,
so no signal is ever decisive: a missing vehicle detection lowers confidence in
occupancy but can never veto it, which matters because a motorcycle behind a van
is invisible to the detector and obvious to the crop classifier.

The uncertain band was chosen by sweeping the error target and taking the knee;
tighter targets make the band asymmetric and drive false-vacant up sharply:

| Target error | Band | Uncertain | Error where decided | False vacant | False occupied |
|---:|---|---:|---:|---:|---:|
| 0.010 | 0.49–0.51 | 0.01% | 0.52% | 0.42% | 0.68% |
| **0.004** | **0.35–0.56** | **0.20%** | **0.40%** | 0.50% | 0.61% |
| 0.003 | 0.45–0.88 | 0.71% | 0.30% | **1.21%** | 0.23% |
| 0.002 | 0.35–0.97 | 1.71% | 0.20% | **2.72%** | 0.12% |

Result at the selected band, confirmed out-of-fold (5-fold, coefficients never
score the fold they were fitted on):

| Signal | Balanced accuracy | False vacant | False occupied |
|---|---:|---:|---:|
| Classifier only | 0.9941 | 0.0061 | 0.0056 |
| Fused (out-of-fold) | 0.9943 | **0.0049** | 0.0065 |

**Honest reading:** fusion reduces the false-vacant rate by 20% relative — the
error that matters most for a parking product — at a small cost in
false-occupied, with balanced accuracy essentially unchanged. On a corpus of
well-lit cars with ground-truth geometry this is the expected magnitude. The
larger value of vehicle detection is scene *coverage*, not flipping bay
verdicts.

### 2.9 Automatic calibration replaces human correction

`backend/app/services/auto_calibration.py`. A fixed camera calibrates itself
from several frames spread across the clip — spread, not consecutive, because
neighbouring frames of a fixed camera would agree about a spurious detection as
readily as about a real bay. Detections that recur are kept and averaged;
transients are dropped.

Measured on a reconstructed unseen-camera sequence (CNRPark camera7, 14 frames):

| Frames | State | Spaces | Consensus | Seconds |
|---:|---|---:|---:|---:|
| 3 | active | 54 | 0.926 | 1.7 |
| 5 | active | 53 | 0.902 | 10.8 |
| 7 | active | 54 | 0.921 | 16.1 |
| 9 | active | 53 | 0.883 | 22.3 |
| 12 | active | 50 | 0.887 | 29.5 |

Stability was measured by calibrating twice from **disjoint halves** of the
sequence, which is a stronger claim than repeatability on one sample because it
shows the layout is a property of the camera rather than of the frames chosen:

| Sequence | Spaces (half A / half B) | Recovered by both at IoU 0.5 | Median polygon IoU |
|---|---|---:|---:|
| CNRPark camera7 | 51 / 49 | **90.2%** | **0.941** |
| PKLot UFPR04 | 8 / 6 | 62.5% | 0.804 |

**And the failure this exposes, stated plainly.** UFPR04 reaches `active` with
8 spaces in a lot that has 28. Calibration is stable on what it finds and
*silently under-covers*: consensus measures agreement among the bays found, not
completeness, so a confident layout can still be a third of a car park. The
signal that reveals it is the parking-area report — a high count of unmapped
vehicles inside the facility against few mapped ones means the bay map is
incomplete — but that is a diagnostic a reader must interpret, not a gate the
system applies.

Three frames were enough on camera7, and more frames did not improve the result;
the cost is roughly 2.5 seconds per frame of calibration on CPU.

### 2.10 Unseen-camera video: what was and was not tested

`ml/build_development_sequence.py` reconstructs chronological fixed-camera
sequences from `val_unseen` stills. Two were built: PKLot UFPR04 (24 frames) and
CNRPark camera7 (14 frames). The protected holdout was never read.

Both were run through the real video job, and **neither reached the calibration
path**: UFPR04 and camera7 are registered identities for the recall path, so the
layout resolved from the reference frame before calibration was consulted. They
are unseen to the *generalized detector*, not to the whole system, and the
corpus contains no camera that is unseen to both with enough chronological
frames.

The calibration branch is therefore exercised directly in
`backend/tests/test_v35_calibration.py`, on the same frames a job would have
used. What is *not* claimed: an end-to-end video job that onboards a camera the
system has never encountered in any form.

Two further properties of reconstructed sequences must travel with any figure
measured on them. Frames are minutes apart, so temporal smoothing lags
transitions far more than on native footage — on the prepared PKLot time-lapse
the error at a transition frame reached 30 spaces while the system correctly
reported 30 of 40 bays uncertain. And appearance changes so much between frames
that the ORB stability check drops them: 8 of 14 camera7 frames were rejected as
camera motion that had not occurred.

---

## 2.11 Performance, and a defect that was misdiagnosed before it was found

Profiling the pipeline produced numbers that did not match the same models
measured on their own. An earlier pass blamed PyTorch for that gap. **That
attribution was wrong, and the correction is recorded here rather than quietly
replaced**, because the wrong answer is instructive: it was reached by comparing
two processes that differed in more than one way.

### The claim that did not survive testing

The earlier measurement was that the vehicle detector runs at 123 ms in a
process that has not imported torch and 2,103 ms in one that has -- a 17x
penalty attributed to torch and ONNX Runtime contending for CPU.

Tested directly, importing torch costs nothing. Two arms -- one that imports the
module defining the PyTorch models, one that does not -- were run *alternately*
in fresh subprocesses for five rounds, alternating which arm went first, on a
machine whose background load makes any two profiles taken minutes apart
incomparable:

| | torch-free | torch imported | ratio |
|---|---:|---:|---:|
| slot localization, median | 3,772 ms | 3,847 ms | 1.02x |
| vehicle detection, median | 2,053 ms | 2,113 ms | 1.03x |

A 17x effect does not hide inside a 1.02x ratio. What the original comparison
actually contained was two differences at once: torch *imported*, and a
torchvision transform *executing* per bay crop between the ONNX calls around it.
The executing transform was the real cost, and removing it is what produced the
gain that was then credited to the import.

### The defect that was actually there

Splitting one detector call apart located the time immediately:

| Part of one slot-localization call | |
|---|---:|
| letterbox and normalise | 16 ms |
| **ONNX session run** | **66 ms** |
| decode 1,002 candidate quadrilaterals | 97 ms |
| polygon NMS | 114 ms |

Sixty-six milliseconds -- with one session loaded. The same graph, on the same
input, in a process where the other three sessions the request path builds are
also alive, takes **832 ms**.

ONNX Runtime sizes each session's intra-op thread pool to every logical
processor it can see. That is correct for a process serving one model. Answering
one image request here keeps four sessions alive -- layout classifier, space
detector, vehicle detector, occupancy classifier -- so on this 16-core hybrid
part it is roughly 88 threads contending for 22 hardware threads, and they
spend their time descheduling one another.

Capping the pool fixes it. Measured over the full detect-and-classify pipeline,
three passes with the order reversed between them so machine drift could not
favour whichever setting ran first:

| Threads per session | Pipeline |
|---|---:|
| ONNX Runtime default | 7,936 ms |
| 8 | 3,818 ms |
| 6 | 2,698 ms |
| 4 | 1,539 ms |
| **3 (shipped)** | **1,062 ms** |
| 2 | 1,172 ms |

The curve is flat between two and three and climbs steeply above four, so the
value sits at the floor rather than on a cliff edge. `PARKING_ONNX_THREADS`
overrides it, and `0` restores ONNX Runtime's own default for a deployment that
serves one model per process.

### What each change was actually worth

Both changes shipped. Their effects are separable because the thread cap is one
environment variable, so the middle column below is the refactored code running
with ONNX Runtime's default pools:

| Stage | torch + ORT default | torch-free + ORT default | torch-free + 3 threads |
|---|---:|---:|---:|
| slot localization | 5,054 ms | 4,534 ms | **731 ms** |
| vehicle detection | 2,957 ms | 3,205 ms | **550 ms** |
| parking-area inference | 113 ms | 132 ms | **20 ms** |
| slot rectification | 593 ms | 299 ms | **119 ms** |
| occupancy classification | 689 ms | 418 ms | **63 ms** |
| association + fusion | 73 ms | 34 ms | **17 ms** |
| **stage total** | **9,478 ms** | **8,621 ms** | **1,499 ms** |
| application import | 10,077 ms | 1,105 ms | **1,086 ms** |
| product-mode request | 11,396 ms | 9,409 ms | **2,517 ms** |
| benchmark-mode request | 2,238 ms | 1,446 ms | **298 ms** |

Read honestly: **removing PyTorch from the serving process is a startup and
footprint win, not a hot-path win.** It takes application import from 10.1 s to
1.1 s and 1,233 loaded modules to none, which is what an operator waits through
on every restart and every worker respawn. It moved the inference stages by
about a tenth -- within this machine's noise. The 6.3x on the hot path came from
the thread cap, and that defect would still be there had the refactor been
shipped on the strength of the original diagnosis.

Absolute figures here are worse than the first isolated measurements of 102 ms
and 123 ms. Those were taken on a quiet machine; these were taken on a laptop
running a browser and several other agents, with hybrid P-cores and E-cores the
scheduler moves work between. Ratios within a table are trustworthy because both
arms ran under the same conditions; the absolute numbers are an upper bound.

GPU is still not reported as a deployment: the installed ONNX Runtime exposes
only a CPU provider, so there is no GPU path through the product to measure.
Detector inference on the RTX 4070 through PyTorch runs at 152-165 ms per image,
which says what the hardware could do, not what the product does.

### The serving/training boundary, and how it is held

Three modules defined PyTorch models beside the code that serves them. They are
now split, serving module and training module, with the artifact-facing half
free of the framework:

| Serving (no torch) | Training and export (torch) |
|---|---|
| `app/ml/occupancy_v3.py` | `app/ml/occupancy_v3_training.py` |
| `app/ml/localization.py` | `app/ml/localization_training.py` |
| `app/ml/template_localizer.py` | `app/ml/template_localizer_training.py` |
| `app/ml/preprocessing.py` (new) | — |

`app/ml/preprocessing.py` holds the resize-scale-normalise pipeline as numpy.
It is asserted *equal* to torchvision, not close to it: exact array equality at
both input sizes, over images from 37x41 to 720x1280.

Two of these transforms were running on the request path -- one per bay crop in
the occupancy classifier, one per image in the layout classifier. Neither
remains.

The boundary is enforced by test, because it is the kind of property that decays
silently: one convenient import in a service would restore it without breaking
any functional test. `tests/test_v35_serving_runtime.py` imports each of ten
serving modules in its own interpreter and fails if torch, torchvision or
ultralytics appears in `sys.modules` -- and separately asserts that the three
training modules *do* still carry torch, so the split cannot be "fixed" by
breaking retraining.

### What the changes did to the numbers

Nothing, to three decimal places, and nothing at all to any decision.

Removing torch is exactly output-preserving: with `PARKING_ONNX_THREADS=0` the
refactored code reproduces the pre-refactor results **byte for byte** across 30
scenarios in both modes -- 1,173 product bays, 1,529 benchmark bays, 9,384
corner coordinates, 5,865 confidence values, 711 vehicle detections. Same
SHA-256.

The thread cap does perturb arithmetic, because ONNX Runtime sums partial
results in a different order. That perturbation was measured rather than
assumed, on 120 images across both samples:

| | |
|---|---:|
| bays compared | 3,925 |
| vehicles compared | 2,676 |
| occupancy verdicts changed | **0** |
| vehicle classes changed | **0** |
| count, placement or benchmark totals changed | **0** |
| largest corner coordinate shift | 8e-08 (1e-04 px) |
| largest vehicle box shift | 1e-06 |
| largest occupancy probability shift | 3e-03 |

Benchmark-mode payloads are byte-identical either way, because benchmark mode
scores ground-truth polygons that no detector output can move. The probability
shift exists only in product mode, and only because a corner that moves by 1e-7
changes which pixels a rectified crop samples. Inference remains deterministic:
two runs at the same setting produce identical bytes.

## 3. Limitations

Stated as findings, not caveats: each one is something a reader could otherwise
assume the numbers cover.

### 3.1 Two-wheeler support is architectural, not validated

The taxonomy carries `TWO_WHEELER`, the mapping is unit-tested, and the runtime
will report one. **No figure in this work establishes that it works in a car
park.** The parking corpus produced roughly one two-wheeler per thousand
detections, which is not a sample. A separate general-domain development set
(COCO val2017, kept outside the parking protocol under `vehicle-dev/`) gives
per-class figures for the detector's *class ability*, but COCO is street
photography: a motorcycle there is metres from the lens, and a motorcycle in a
car park is thirty metres below a CCTV mast. Good numbers there do not transfer.

Status: **supported by architecture, not sufficiently validated.**

### 3.2 Neither space model is good enough on unseen cameras

The best unseen recall@50 measured is 0.215, and at the strict IoU 0.75 it is
0.025. The binding constraint is viewpoint diversity (§1.5), not architecture:
one PKLot site, nine CNRPark cameras and 227 single-image ACPDS captures. Every
run peaks within the first handful of epochs and then degrades, which is what
fitting a handful of viewpoints looks like.

### 3.3 Calibration can under-cover without knowing it

Consensus measures agreement among the bays found, not completeness. On UFPR04
the system establishes a confident, stable layout of 8 spaces in a 28-space car
park (§2.9). The unmapped-vehicle count is the diagnostic that reveals it; there
is no gate that acts on it.

### 3.4 PKLot annotations describe car-standing regions, not bay boundaries

Measured directly: bay edges carry 0.91× the marking response of ground
displaced away from them, against ACPDS's 2.27×. This penalises a model that
predicts true geometry on 86% of the evaluation bays, makes marking refinement
unusable (§2.2), and means the aggregate geometry numbers understate what the
four-corner model does on geometry-faithful data. It is why §2.1 reports ACPDS
separately rather than only in aggregate.

### 3.5 The parking area can over-reach

Because parked vehicles seed the region, a dense stationary queue on an adjacent
road could merge into the facility. What prevents it is a kerb-width gap larger
than the closing radius — a property of the site, not a guarantee.

### 3.6 No camera is unseen to the whole system in video

Both reconstructed sequences resolve through the registered-camera path (§2.10).
End-to-end onboarding of a wholly unfamiliar camera in a video job is untested.

### 3.7 Fusion is fitted against ground-truth polygons

The coefficients were fitted with dataset bay geometry, because a bay the
detector never proposed has no label to fit against. Production runs on detected
polygons, where geometry is worse and the crop the classifier sees is
correspondingly worse.

### 3.8 Vehicle figures come from a COCO-pretrained detector

It is not fine-tuned on parking imagery, and it fails in a specific,
reproducible way: on steep top-down views it labels ordinary cars `bus` with
0.33 confidence and `car` with 0.007. Those objects are kept as unclassified so
they still inform occupancy, but the system cannot name them, and on such views
the class counts will understate the real number of cars.
