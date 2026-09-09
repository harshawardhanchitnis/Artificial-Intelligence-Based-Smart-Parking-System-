"""Full-scene vehicle detection, reported in the three product classes.

The occupancy classifier only ever sees a crop of a bay the system already
knows about, so a vehicle standing anywhere the bay detector missed does not
exist as far as the product is concerned.  Measured on unseen cameras, the bay
detector misses 57% of vacant bays and 29% of occupied ones, which means a
sizeable share of clearly visible vehicles were invisible to the whole system.

This detector runs on the whole frame instead, independently of bay geometry,
so a vehicle is found whether or not its bay was.  That independence is the
point: it is what lets the system notice that its own parking-space map is
incomplete, and what supplies the second opinion the occupancy fusion needs.

Detections are folded into the three product classes at the boundary here, so
no other part of the system ever sees a source taxonomy.  Inference is ONNX
Runtime on the same CPU path as everything else.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.ml import registry
from app.ml.generalized_localizer import letterbox
from app.ml.registry import inference_providers, session_options
from app.ml.vehicle_taxonomy import (
    PRODUCT_CLASSES,
    UNSUPPORTED_VEHICLE_LABELS,
    product_class,
)

DETECTOR_NAME = "vehicle-detector-v3"
DETECTOR_ONNX = f"{DETECTOR_NAME}.onnx"
DETECTOR_METADATA = f"{DETECTOR_NAME}.json"
DETECTOR_SCHEMA = "1.0"


class VehicleDetectorNotReadyError(RuntimeError):
    """Raised when the vehicle detector is unavailable."""


@dataclass(frozen=True)
class VehicleDetection:
    """One detected object, with a product class only if it has one.

    ``product_class`` is ``None`` for something the detector recognises as a
    vehicle but the product taxonomy does not cover -- a coach, a bicycle -- and
    also for the case that turns out to matter more: a detection the model is
    confident about under an unsupported label and unconfident about under every
    supported one.

    Such an object is still real, and still fills a bay.  It therefore counts as
    occupancy evidence exactly like a classified vehicle, and is excluded only
    from the class counts, so a coach can never appear as a lorry and a
    misclassified car can never appear as anything at all.
    """

    product_class: str | None
    confidence: float
    box: tuple[float, float, float, float]
    source_label: str

    @property
    def classified(self) -> bool:
        return self.product_class is not None

    def as_dict(self) -> dict[str, object]:
        x1, y1, x2, y2 = self.box
        return {
            "vehicle_class": self.product_class,
            "confidence": round(self.confidence, 6),
            "box": [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)],
        }


@dataclass(frozen=True)
class SceneDiagnostics:
    """What the detector was confident about, whether or not it was a vehicle.

    A bare "0 cars" is ambiguous: it can mean an empty car park, or a detector
    that cannot read this viewpoint.  These counts separate the two.  Measured
    on a near-vertical aerial view, the detector returns eighteen confident
    objects of which none are vehicle-shaped -- it labels cars ``cell phone``
    and ``sink`` -- while the same threshold on an oblique view of the same car
    park returns dozens of cars.  Reporting the confident non-vehicle count lets
    the interface say which situation it is in without guessing.
    """

    confident_objects: int
    vehicle_shaped_objects: int
    top_non_vehicle_labels: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "confident_objects": self.confident_objects,
            "vehicle_shaped_objects": self.vehicle_shaped_objects,
            "top_non_vehicle_labels": list(self.top_non_vehicle_labels),
        }


def non_max_suppression(
    boxes: np.ndarray, scores: np.ndarray, threshold: float
) -> list[int]:
    """Greedy suppression over axis-aligned boxes, highest score first."""
    order = np.argsort(-scores)
    areas = np.maximum(boxes[:, 2] - boxes[:, 0], 0) * np.maximum(boxes[:, 3] - boxes[:, 1], 0)
    kept: list[int] = []
    while order.size:
        current = int(order[0])
        kept.append(current)
        if order.size == 1:
            break
        rest = order[1:]
        left = np.maximum(boxes[current, 0], boxes[rest, 0])
        top = np.maximum(boxes[current, 1], boxes[rest, 1])
        right = np.minimum(boxes[current, 2], boxes[rest, 2])
        bottom = np.minimum(boxes[current, 3], boxes[rest, 3])
        overlap = np.maximum(right - left, 0) * np.maximum(bottom - top, 0)
        union = areas[current] + areas[rest] - overlap
        order = rest[np.where(union > 0, overlap / np.maximum(union, 1e-9), 0.0) < threshold]
    return kept


@dataclass
class VehicleDetector:
    session: ort.InferenceSession
    metadata: dict[str, object]
    input_size: int
    confidence_threshold: float
    iou_threshold: float
    max_detections: int
    class_names: list[str]

    @classmethod
    def load(cls, model_root: Path) -> VehicleDetector:
        """Return the shared detector, built once per model file version."""
        model_path = model_root / DETECTOR_ONNX
        metadata_path = model_root / DETECTOR_METADATA
        signature = registry.file_signature(model_path, metadata_path)
        cached_error = registry.cached_failure("vehicle_detector", signature)
        if cached_error is not None:
            raise cached_error

        def build() -> VehicleDetector:
            if not model_path.is_file() or not metadata_path.is_file():
                raise VehicleDetectorNotReadyError("Vehicle detector is not installed")
            metadata = registry.read_metadata(metadata_path)
            if metadata.get("schema_version") != DETECTOR_SCHEMA:
                raise VehicleDetectorNotReadyError("Vehicle detector schema is unsupported")
            names = metadata.get("class_names")
            if not isinstance(names, list) or not names:
                raise VehicleDetectorNotReadyError("Vehicle detector metadata lists no classes")
            return cls(
                session=ort.InferenceSession(
                    str(model_path),
                    sess_options=session_options(),
                    providers=inference_providers(),
                ),
                metadata=metadata,
                input_size=int(metadata.get("input_size", 960)),
                confidence_threshold=float(metadata.get("confidence_threshold", 0.30)),
                iou_threshold=float(metadata.get("iou_threshold", 0.50)),
                max_detections=int(metadata.get("max_detections", 300)),
                class_names=[str(name) for name in names],
            )

        try:
            return registry.cached("vehicle_detector", signature, build)
        except VehicleDetectorNotReadyError as exc:
            registry.remember_failure("vehicle_detector", signature, exc)
            raise

    def detect(
        self, image: Image.Image, *, confidence: float | None = None
    ) -> list[VehicleDetection]:
        """Detect supported vehicles across the whole frame.

        Boxes come back in normalised image coordinates so they can be compared
        against bay polygons, which are stored the same way.
        """
        return self.detect_with_diagnostics(image, confidence=confidence)[0]

    def detect_with_diagnostics(
        self, image: Image.Image, *, confidence: float | None = None
    ) -> tuple[list[VehicleDetection], SceneDiagnostics]:
        """The same single inference pass, also reporting what it rejected."""
        threshold = self.confidence_threshold if confidence is None else confidence
        canvas, scale, pad_x, pad_y = letterbox(image, self.input_size)
        batch = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0

        raw = self.session.run(None, {self.session.get_inputs()[0].name: batch})[0]
        predictions = np.asarray(raw, dtype=np.float32)
        if predictions.ndim == 3:
            predictions = predictions[0]
        # Ultralytics emits (4 + classes, anchors); orient so each row is a box.
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.transpose()
        if predictions.shape[1] < 5:
            return [], SceneDiagnostics(0, 0, ())

        class_scores = predictions[:, 4 : 4 + len(self.class_names)]
        best_class = np.argmax(class_scores, axis=1)
        best_score = class_scores[np.arange(class_scores.shape[0]), best_class]

        # Everything vehicle-shaped is kept; the taxonomy is applied afterwards.
        #
        # Ranking within the supported classes instead was tried and rejected on
        # measurement.  On a steep top-down PKLot view the detector scores an
        # ordinary car ``bus`` 0.33 and ``car`` 0.007, so no rule over the
        # model's own probabilities recovers it as a car -- the detector is
        # simply wrong there, and confidently so.  Ranking within supported
        # classes therefore deleted 139 real objects from one frame.  Keeping
        # them unclassified preserves the occupancy evidence they carry while
        # denying them a class the model has not earned.
        vehicle_like = np.array(
            [
                product_class(name) is not None or name.lower() in UNSUPPORTED_VEHICLE_LABELS
                for name in self.class_names
            ],
            dtype=bool,
        )
        keep = (best_score >= threshold) & vehicle_like[best_class]

        confident = best_score >= threshold
        rejected = confident & ~vehicle_like[best_class]
        labels: list[str] = []
        for index in np.argsort(-best_score[rejected])[:3]:
            name = self.class_names[int(best_class[rejected][index])]
            if name not in labels:
                labels.append(name)
        diagnostics = SceneDiagnostics(
            confident_objects=int(confident.sum()),
            vehicle_shaped_objects=int(keep.sum()),
            top_non_vehicle_labels=tuple(labels),
        )

        if not np.any(keep):
            return [], diagnostics

        selected = predictions[keep]
        scores = best_score[keep]
        classes = best_class[keep]

        centres = selected[:, :2]
        sizes = selected[:, 2:4]
        corners = np.concatenate([centres - sizes / 2, centres + sizes / 2], axis=1)

        detections: list[VehicleDetection] = []
        for index in non_max_suppression(corners, scores, self.iou_threshold)[
            : self.max_detections
        ]:
            x1, y1, x2, y2 = corners[index]
            box = (
                float(np.clip((x1 - pad_x) / scale / image.width, 0.0, 1.0)),
                float(np.clip((y1 - pad_y) / scale / image.height, 0.0, 1.0)),
                float(np.clip((x2 - pad_x) / scale / image.width, 0.0, 1.0)),
                float(np.clip((y2 - pad_y) / scale / image.height, 0.0, 1.0)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            source_label = self.class_names[int(classes[index])]
            mapped = product_class(source_label)
            detections.append(
                VehicleDetection(
                    product_class=mapped,
                    confidence=float(scores[index]),
                    box=box,
                    source_label=source_label,
                )
            )
        return detections, diagnostics


def vehicle_counts(detections: list[VehicleDetection]) -> dict[str, int]:
    """Count detections per product class, always with all three keys present.

    Unclassified objects are deliberately absent: they are evidence, not a
    fourth vehicle type, and must never reach a user-facing class count.
    """
    counts = dict.fromkeys(PRODUCT_CLASSES, 0)
    for detection in detections:
        if detection.product_class is not None:
            counts[detection.product_class] += 1
    return counts


def unclassified_count(detections: list[VehicleDetection]) -> int:
    """Vehicle-shaped objects the taxonomy does not cover, for diagnostics."""
    return sum(1 for detection in detections if detection.product_class is None)


def detector_status(model_root: Path) -> dict[str, object]:
    """Report readiness from metadata alone, without building a session."""
    metadata_path = model_root / DETECTOR_METADATA
    model_path = model_root / DETECTOR_ONNX
    if not metadata_path.is_file() or not model_path.is_file():
        return {
            "ready": False,
            "model_name": DETECTOR_NAME,
            "reason": "Vehicle detector is not installed",
        }
    try:
        metadata = registry.read_metadata(metadata_path)
    except (OSError, json.JSONDecodeError) as exc:
        return {"ready": False, "model_name": DETECTOR_NAME, "reason": type(exc).__name__}
    return {
        "ready": metadata.get("schema_version") == DETECTOR_SCHEMA,
        "model_name": DETECTOR_NAME,
        "architecture": metadata.get("architecture"),
        "input_size": metadata.get("input_size"),
        "confidence_threshold": metadata.get("confidence_threshold"),
        "product_classes": list(PRODUCT_CLASSES),
        "source_taxonomy": metadata.get("source_taxonomy"),
        "development": metadata.get("development"),
        "trained_at": metadata.get("trained_at"),
    }
