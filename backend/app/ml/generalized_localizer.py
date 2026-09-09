"""Generalized parking-space detection for cameras the system has never seen.

The V2 localizer classified an image into one of twelve registered camera
identities and returned that camera's stored polygons.  It had no way to say
"I do not recognise this lot", so an unfamiliar view was answered with another
camera's geometry.

This module replaces that with an oriented-bounding-box detector that predicts
parking-space quadrilaterals from pixels.  Because it detects rather than
recalls, an unfamiliar lot produces its own geometry, and an image with no
parking spaces produces no detections instead of a confident wrong answer.

Inference runs through ONNX Runtime on the same CPU path as the rest of the
application; the detector is trained separately on a CUDA machine.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

from app.ml import registry
from app.ml.geometry import order_polygon, polygon_iou, validate_polygon
from app.ml.registry import inference_providers, session_options

DETECTOR_NAME = "parking-space-detector-v3"
DETECTOR_ONNX = f"{DETECTOR_NAME}.onnx"
DETECTOR_METADATA = f"{DETECTOR_NAME}.json"
DETECTOR_SCHEMA = "1.0"

# How a detector expresses a bay.  ``obb`` regresses a rotated rectangle;
# ``pose`` regresses four independent corners and can therefore follow a
# perspective-projected bay, which a rectangle cannot.  Both are decoded
# here so a model can be swapped by reinstalling an artifact, without the
# rest of the system knowing which one it is.
REPRESENTATION_OBB = "obb"
REPRESENTATION_POSE = "pose"


class DetectorNotReadyError(RuntimeError):
    """Raised when the generalized parking-space detector is unavailable."""


def letterbox(image: Image.Image, size: int) -> tuple[np.ndarray, float, float, float]:
    """Resize into a square canvas without distorting aspect ratio.

    Returns the canvas plus the scale and padding needed to map detections back
    to normalised source coordinates.
    """
    scale = min(size / image.width, size / image.height)
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))
    resized = image.convert("RGB").resize((width, height), Image.BILINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    pad_x, pad_y = (size - width) // 2, (size - height) // 2
    canvas[pad_y : pad_y + height, pad_x : pad_x + width] = np.asarray(resized)
    return canvas, scale, float(pad_x), float(pad_y)


def rotated_box_corners(
    center_x: float, center_y: float, width: float, height: float, angle: float
) -> list[list[float]]:
    """Four corners of a rotated box, in image pixels."""
    cos_a, sin_a = float(np.cos(angle)), float(np.sin(angle))
    half_w, half_h = width / 2.0, height / 2.0
    corners = []
    for dx, dy in ((-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)):
        corners.append(
            [center_x + dx * cos_a - dy * sin_a, center_y + dx * sin_a + dy * cos_a]
        )
    return corners


def polygon_nms(
    detections: list[dict[str, object]], iou_threshold: float, limit: int
) -> list[dict[str, object]]:
    """Greedy non-maximum suppression over rotated quadrilaterals.

    Exact polygon overlap is expensive and a dense car park produces hundreds of
    candidates, so each pair is first tested with axis-aligned bounding boxes.
    Two quadrilaterals whose bounding boxes do not overlap cannot overlap, which
    removes almost every pair before the exact test runs.
    """
    ordered = sorted(detections, key=lambda item: float(item["confidence"]), reverse=True)
    bounds = [
        (
            min(x for x, _ in item["polygon"]),  # type: ignore[union-attr]
            min(y for _, y in item["polygon"]),  # type: ignore[union-attr]
            max(x for x, _ in item["polygon"]),  # type: ignore[union-attr]
            max(y for _, y in item["polygon"]),  # type: ignore[union-attr]
        )
        for item in ordered
    ]
    kept: list[dict[str, object]] = []
    kept_bounds: list[tuple[float, float, float, float]] = []
    for candidate, box in zip(ordered, bounds, strict=True):
        suppressed = False
        for other, other_box in zip(kept, kept_bounds, strict=True):
            if (
                box[0] >= other_box[2]
                or box[2] <= other_box[0]
                or box[1] >= other_box[3]
                or box[3] <= other_box[1]
            ):
                continue
            if polygon_iou(candidate["polygon"], other["polygon"]) >= iou_threshold:  # type: ignore[arg-type]
                suppressed = True
                break
        if suppressed:
            continue
        kept.append(candidate)
        kept_bounds.append(box)
        if len(kept) >= limit:
            break
    return kept


@dataclass
class GeneralizedDetector:
    session: ort.InferenceSession
    metadata: dict[str, object]
    input_size: int
    confidence_threshold: float
    iou_threshold: float
    max_detections: int
    representation: str = REPRESENTATION_OBB

    @classmethod
    def load(cls, model_root: Path) -> GeneralizedDetector:
        """Return the shared detector, built once per model file version."""
        model_path = model_root / DETECTOR_ONNX
        metadata_path = model_root / DETECTOR_METADATA
        signature = registry.file_signature(model_path, metadata_path)
        cached_error = registry.cached_failure("space_detector", signature)
        if cached_error is not None:
            raise cached_error

        def build() -> GeneralizedDetector:
            if not model_path.is_file() or not metadata_path.is_file():
                raise DetectorNotReadyError(
                    "Generalized parking-space detector is not installed"
                )
            metadata = registry.read_metadata(metadata_path)
            if metadata.get("schema_version") != DETECTOR_SCHEMA:
                raise DetectorNotReadyError("Generalized detector schema is unsupported")
            return cls(
                session=ort.InferenceSession(
                    str(model_path),
                    sess_options=session_options(),
                    providers=inference_providers(),
                ),
                metadata=metadata,
                input_size=int(metadata.get("input_size", 1024)),
                confidence_threshold=float(metadata.get("confidence_threshold", 0.25)),
                iou_threshold=float(metadata.get("iou_threshold", 0.30)),
                max_detections=int(metadata.get("max_detections", 400)),
                representation=str(
                    metadata.get("representation", REPRESENTATION_OBB)
                ),
            )

        try:
            return registry.cached("space_detector", signature, build)
        except DetectorNotReadyError as exc:
            registry.remember_failure("space_detector", signature, exc)
            raise

    def detect(
        self, image: Image.Image, *, confidence: float | None = None
    ) -> list[dict[str, object]]:
        """Detect parking-space quadrilaterals, in normalised image coordinates."""
        threshold = self.confidence_threshold if confidence is None else confidence
        canvas, scale, pad_x, pad_y = letterbox(image, self.input_size)
        batch = canvas.astype(np.float32).transpose(2, 0, 1)[None] / 255.0

        raw = self.session.run(None, {self.session.get_inputs()[0].name: batch})[0]
        predictions = np.asarray(raw, dtype=np.float32)
        if predictions.ndim == 3:
            predictions = predictions[0]
        # Ultralytics emits (channels, anchors); orient so each row is one box.
        if predictions.shape[0] < predictions.shape[1]:
            predictions = predictions.transpose()
        minimum_channels = 17 if self.representation == REPRESENTATION_POSE else 6
        if predictions.shape[1] < minimum_channels:
            return []

        scores = predictions[:, 4]
        keep = scores >= threshold
        if not np.any(keep):
            return []
        selected = predictions[keep]
        scores = scores[keep]

        detections: list[dict[str, object]] = []
        for row, score in zip(selected, scores, strict=True):
            corners = self._corners(row)
            polygon = []
            for x, y in corners:
                # Undo letterbox padding and scaling, then normalise.
                source_x = (x - pad_x) / scale / image.width
                source_y = (y - pad_y) / scale / image.height
                polygon.append([min(max(source_x, 0.0), 1.0), min(max(source_y, 0.0), 1.0)])
            polygon = order_polygon(polygon)
            if not validate_polygon(polygon).valid:
                continue
            detections.append(
                {
                    "polygon": polygon,
                    "confidence": round(float(score), 6),
                    "localization_confidence": round(float(score), 6),
                    "corner_confidence": round(float(score), 6),
                }
            )
        return polygon_nms(detections, self.iou_threshold, self.max_detections)

    def _corners(self, row: np.ndarray) -> list[list[float]]:
        """The four corners this detector predicts, in canvas pixels.

        A pose head emits ``cx cy w h obj`` followed by ``x y visibility``
        per keypoint.  The visibility channel is ignored: a bay corner is a
        property of the ground and stays where it is whether or not a
        parked vehicle happens to hide it.
        """
        if self.representation == REPRESENTATION_POSE:
            return [
                [float(row[5 + index * 3]), float(row[6 + index * 3])]
                for index in range(4)
            ]
        return rotated_box_corners(
            float(row[0]), float(row[1]), float(row[2]), float(row[3]), float(row[-1])
        )


def detector_status(model_root: Path) -> dict[str, object]:
    """Report detector readiness from metadata alone, without a session."""
    metadata_path = model_root / DETECTOR_METADATA
    model_path = model_root / DETECTOR_ONNX
    if not metadata_path.is_file() or not model_path.is_file():
        return {
            "ready": False,
            "model_name": DETECTOR_NAME,
            "reason": "Generalized parking-space detector is not installed",
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
        "representation": metadata.get("representation", REPRESENTATION_OBB),
        "development": metadata.get("development"),
        "trained_at": metadata.get("trained_at"),
        "moving_camera_supported": False,
    }
