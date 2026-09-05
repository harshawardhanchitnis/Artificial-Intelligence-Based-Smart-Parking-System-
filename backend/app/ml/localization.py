from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from app.ml.geometry import order_polygon, polygon_iou, validate_polygon
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


class ParkingSlotLocalizer(nn.Module):
    """One-stage oriented-slot proposal network: confidence plus four normalized corners."""

    def __init__(self, *, pretrained: bool = False) -> None:
        super().__init__()
        backbone = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
        self.features = backbone.features
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(576, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.SiLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
        )
        self.head = nn.Conv2d(32, 9, 1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.upsample(self.features(inputs)))


def localizer_preprocessing(training: bool = False) -> transforms.Compose:
    operations: list[object] = [transforms.Resize((INPUT_HEIGHT, INPUT_WIDTH))]
    if training:
        operations.extend(
            [
                transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
                transforms.RandomAutocontrast(p=0.15),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(operations)  # type: ignore[arg-type]


def localizer_targets(
    slots: list[dict[str, object]], height: int, width: int
) -> tuple[torch.Tensor, torch.Tensor, int]:
    objectness = torch.zeros(1, height, width)
    corners = torch.zeros(8, height, width)
    collisions = 0
    for slot in slots:
        polygon = order_polygon(slot["polygon"])  # type: ignore[arg-type]
        center_x = sum(point[0] for point in polygon) / 4
        center_y = sum(point[1] for point in polygon) / 4
        x = min(width - 1, max(0, int(center_x * width)))
        y = min(height - 1, max(0, int(center_y * height)))
        if objectness[0, y, x] > 0:
            collisions += 1
            continue
        objectness[0, y, x] = 1
        center = torch.tensor([(x + 0.5) / width, (y + 0.5) / height])
        points = torch.tensor(polygon, dtype=torch.float32)
        corners[:, y, x] = ((points - center) / MAX_CORNER_OFFSET).clamp(-1, 1).reshape(-1)
    return objectness, corners, collisions


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_localizer(
    model: ParkingSlotLocalizer, model_root: Path, metadata: dict[str, object]
) -> dict[str, object]:
    model_root.mkdir(parents=True, exist_ok=True)
    destination = model_root / LOCALIZER_ONNX
    temporary = destination.with_suffix(".onnx.tmp")
    model.eval().cpu()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, INPUT_HEIGHT, INPUT_WIDTH),
        temporary,
        input_names=["image"],
        output_names=["slot_proposals"],
        dynamic_axes={"image": {0: "batch"}, "slot_proposals": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    os.replace(temporary, destination)
    payload = {
        **metadata,
        "schema_version": LOCALIZER_SCHEMA,
        "model_name": LOCALIZER_NAME,
        "input_width": INPUT_WIDTH,
        "input_height": INPUT_HEIGHT,
        "onnx_file": LOCALIZER_ONNX,
        "onnx_sha256": _sha256(destination),
        "primary_mode": "fully automatic",
        "moving_camera_supported": False,
    }
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=model_root, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
        temporary_metadata = Path(handle.name)
    os.replace(temporary_metadata, model_root / LOCALIZER_METADATA)
    return payload


@dataclass
class SlotLocalizerPredictor:
    session: ort.InferenceSession
    metadata: dict[str, object]
    object_threshold: float
    layout_classifier: LayoutClassifierPredictor

    @classmethod
    def load(cls, model_root: Path) -> SlotLocalizerPredictor:
        model_path = model_root / LOCALIZER_ONNX
        metadata_path = model_root / LOCALIZER_METADATA
        if not model_path.is_file() or not metadata_path.is_file():
            raise LocalizerNotReadyError("Automatic parking-space localizer is not installed")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("schema_version") != LOCALIZER_SCHEMA:
            raise LocalizerNotReadyError("Automatic localizer schema is unsupported")
        if metadata.get("onnx_sha256") != _sha256(model_path):
            raise LocalizerNotReadyError("Automatic localizer checksum is invalid")
        if "final_holdout" not in metadata:
            raise LocalizerNotReadyError("Automatic localizer has not passed its holdout gate")
        try:
            layout_classifier = LayoutClassifierPredictor.load(model_root)
        except RuntimeError as exc:
            raise LocalizerNotReadyError(str(exc)) from exc
        return cls(
            ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"]),
            metadata,
            float(metadata["object_threshold"]),
            layout_classifier,
        )

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
    try:
        predictor = SlotLocalizerPredictor.load(model_root)
    except (LocalizerNotReadyError, OSError, json.JSONDecodeError) as exc:
        return {"ready": False, "model_name": LOCALIZER_NAME, "reason": str(exc)}
    return {
        "ready": True,
        "model_name": LOCALIZER_NAME,
        "object_threshold": predictor.object_threshold,
        "localization_strategy": predictor.metadata.get("localization_strategy"),
        "supported_datasets": predictor.metadata.get("supported_datasets", []),
        "excluded_datasets": predictor.metadata.get("excluded_datasets", {}),
        "final_holdout": predictor.metadata.get("final_holdout"),
        "moving_camera_supported": False,
    }
