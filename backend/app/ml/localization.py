"""Serve the V1 automatic slot localizer from its exported artifact.

Request-path code only.  The proposal network, its target encoding and the ONNX
export that produced the installed artifact live in
``app.ml.localization_training``, so the FastAPI application never imports the
training framework to answer a request.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from app.ml import registry
from app.ml.geometry import order_polygon, polygon_iou, validate_polygon
from app.ml.registry import inference_providers, session_options
from app.ml.template_localizer import LayoutClassifierPredictor

LOCALIZER_NAME = "parking-slot-localizer-v1"
LOCALIZER_SCHEMA = "1.0"
LOCALIZER_ONNX = f"{LOCALIZER_NAME}.onnx"
LOCALIZER_METADATA = f"{LOCALIZER_NAME}.json"
INPUT_WIDTH = 512
INPUT_HEIGHT = 384
GRID_WIDTH = 128
GRID_HEIGHT = 96
MAX_CORNER_OFFSET = 0.15


class LocalizerNotReadyError(RuntimeError):
    """Raised when automatic parking-space localisation cannot run."""


def decode_localizer_output(
    output: np.ndarray,
    *,
    object_threshold: float,
    nms_iou: float = 0.3,
    limit: int = 300,
) -> list[dict[str, object]]:
    tensor = np.asarray(output, dtype=np.float32)
    if tensor.ndim == 4:
        tensor = tensor[0]
    confidence = 1.0 / (1.0 + np.exp(-np.clip(tensor[0], -30, 30)))
    pooled = cv2.dilate(confidence, np.ones((3, 3), np.uint8))
    points = np.argwhere((confidence >= object_threshold) & (confidence >= pooled - 1e-7))
    proposals = []
    for y, x in points:
        base = np.asarray([(x + 0.5) / confidence.shape[1], (y + 0.5) / confidence.shape[0]])
        offsets = np.tanh(tensor[1:, y, x]).reshape(4, 2) * MAX_CORNER_OFFSET
        polygon = order_polygon((base + offsets).clip(0, 1).tolist())
        quality = validate_polygon(polygon)
        if not quality.valid:
            continue
        proposals.append(
            {
                "polygon": polygon,
                "localization_confidence": float(confidence[y, x]),
                "corner_confidence": float(confidence[y, x] * min(1.0, quality.area / 0.0008)),
            }
        )
    proposals.sort(key=lambda item: float(item["localization_confidence"]), reverse=True)
    selected = []
    for proposal in proposals:
        if any(
            polygon_iou(proposal["polygon"], kept["polygon"]) >= nms_iou  # type: ignore[arg-type]
            for kept in selected
        ):
            continue
        selected.append(proposal)
        if len(selected) >= limit:
            break
    return selected


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def localizer_metadata(model_root: Path) -> dict[str, object]:
    """Validate and return localizer metadata without building a session."""
    model_path = model_root / LOCALIZER_ONNX
    metadata_path = model_root / LOCALIZER_METADATA
    if not model_path.is_file() or not metadata_path.is_file():
        raise LocalizerNotReadyError("Automatic parking-space localizer is not installed")
    metadata = registry.read_metadata(metadata_path)
    if metadata.get("schema_version") != LOCALIZER_SCHEMA:
        raise LocalizerNotReadyError("Automatic localizer schema is unsupported")
    if "final_holdout" not in metadata:
        raise LocalizerNotReadyError("Automatic localizer has not passed its holdout gate")
    return metadata


@dataclass
class SlotLocalizerPredictor:
    session: ort.InferenceSession
    metadata: dict[str, object]
    object_threshold: float
    layout_classifier: LayoutClassifierPredictor

    @classmethod
    def load(cls, model_root: Path) -> SlotLocalizerPredictor:
        """Return the shared localizer, building it once per model file version."""
        model_path = model_root / LOCALIZER_ONNX
        metadata_path = model_root / LOCALIZER_METADATA
        signature = registry.file_signature(model_path, metadata_path)

        def build() -> SlotLocalizerPredictor:
            metadata = localizer_metadata(model_root)
            digest = registry.cached(
                f"sha256::{model_path}",
                registry.file_signature(model_path),
                lambda: sha256_file(model_path),
            )
            if metadata.get("onnx_sha256") != digest:
                raise LocalizerNotReadyError("Automatic localizer checksum is invalid")
            try:
                layout_classifier = LayoutClassifierPredictor.load(model_root)
            except RuntimeError as exc:
                raise LocalizerNotReadyError(str(exc)) from exc
            return cls(
                ort.InferenceSession(
                    str(model_path),
                    sess_options=session_options(),
                    providers=inference_providers(),
                ),
                metadata,
                float(metadata["object_threshold"]),
                layout_classifier,
            )

        return registry.cached("slot_localizer", signature, build)

    def detect(self, image: Image.Image) -> dict[str, object]:
        result = self.layout_classifier.detect(image)
        slots = result["slots"]
        confidence = float(result["confidence"])
        count_bounds = self.metadata.get("supported_slot_count", {"minimum": 4, "maximum": 250})
        minimum = int(count_bounds.get("minimum", 4))  # type: ignore[union-attr]
        maximum = int(count_bounds.get("maximum", 250))  # type: ignore[union-attr]
        if result["status"] == "unsupported_layout" or len(slots) < minimum or len(slots) > maximum:
            status = "unsupported_layout"
        elif result["status"] == "success_with_warnings":
            status = "success_with_warnings"
        else:
            status = "success"
        return {
            "status": status,
            "confidence": round(confidence, 6),
            "slots": slots,
            "model_name": LOCALIZER_NAME,
            "layout_key": result.get("layout_key"),
            "method": result.get("method"),
        }


def localizer_status(model_root: Path) -> dict[str, object]:
    """Report localizer readiness from metadata alone, without a session."""
    try:
        metadata = localizer_metadata(model_root)
    except (LocalizerNotReadyError, OSError, json.JSONDecodeError) as exc:
        return {"ready": False, "model_name": LOCALIZER_NAME, "reason": str(exc)}
    return {
        "ready": True,
        "model_name": LOCALIZER_NAME,
        "object_threshold": float(metadata["object_threshold"]),
        "localization_strategy": metadata.get("localization_strategy"),
        "supported_datasets": metadata.get("supported_datasets", []),
        "excluded_datasets": metadata.get("excluded_datasets", {}),
        "final_holdout": metadata.get("final_holdout"),
        "moving_camera_supported": False,
    }
