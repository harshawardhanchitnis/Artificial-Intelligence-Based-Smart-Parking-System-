from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

OCCUPANCY_MODEL_NAME = "parking-occupancy-enhanced-v3"
OCCUPANCY_SCHEMA_VERSION = "3.0"
OCCUPANCY_ONNX = f"{OCCUPANCY_MODEL_NAME}.onnx"
OCCUPANCY_METADATA = f"{OCCUPANCY_MODEL_NAME}.json"
IMAGE_SIZE = 128


class OccupancyV3NotReadyError(RuntimeError):
    """Raised when the independently evaluated V3 model is unavailable."""


class CompactOccupancyCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 3, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, 3, padding=1),
            nn.BatchNorm2d(48),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, 3, padding=1),
            nn.BatchNorm2d(96),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(96, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(inputs)).squeeze(1)


class BinaryOutputWrapper(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.model(inputs).squeeze(1)


def build_occupancy_model(architecture: str, *, pretrained: bool = False) -> nn.Module:
    weights = "DEFAULT" if pretrained else None
    if architecture == "compact-cnn":
        return CompactOccupancyCNN()
    if architecture == "mobilenet-v3-small":
        model = models.mobilenet_v3_small(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        return BinaryOutputWrapper(model)
    if architecture == "efficientnet-b0":
        model = models.efficientnet_b0(weights=weights)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        return BinaryOutputWrapper(model)
    raise ValueError(f"Unsupported occupancy architecture: {architecture}")


def preprocessing(training: bool = False) -> transforms.Compose:
    operations: list[object] = [transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))]
    if training:
        operations.extend(
            [
                transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
                transforms.RandomAffine(4, translate=(0.03, 0.03), scale=(0.95, 1.05)),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(operations)  # type: ignore[arg-type]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def export_occupancy_model(
    model: nn.Module,
    model_root: Path,
    *,
    architecture: str,
    threshold: float,
    temperature: float,
    metadata: dict[str, object],
) -> dict[str, object]:
    model_root.mkdir(parents=True, exist_ok=True)
    model.eval().cpu()
    destination = model_root / OCCUPANCY_ONNX
    temporary = destination.with_suffix(".onnx.tmp")
    torch.onnx.export(
        model,
        torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE),
        temporary,
        input_names=["image"],
        output_names=["logit"],
        dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    os.replace(temporary, destination)
    payload = {
        **metadata,
        "schema_version": OCCUPANCY_SCHEMA_VERSION,
        "model_name": OCCUPANCY_MODEL_NAME,
        "architecture": architecture,
        "input_size": IMAGE_SIZE,
        "preprocessing": "perspective-mask-v2-128 + ImageNet normalization",
        "decision_threshold": round(float(threshold), 8),
        "temperature": round(float(temperature), 8),
        "onnx_file": OCCUPANCY_ONNX,
        "onnx_sha256": sha256_file(destination),
        "frozen": True,
    }
    atomic_json(model_root / OCCUPANCY_METADATA, payload)
    return payload


@dataclass
class OccupancyV3Predictor:
    session: ort.InferenceSession
    threshold: float
    temperature: float
    metadata: dict[str, object]

    @classmethod
    def load(cls, model_root: Path) -> OccupancyV3Predictor:
        model_path = model_root / OCCUPANCY_ONNX
        metadata_path = model_root / OCCUPANCY_METADATA
        if not model_path.is_file() or not metadata_path.is_file():
            raise OccupancyV3NotReadyError(
                "Enhanced occupancy model is not installed; "
                "the reproducible V2 baseline remains available"
            )
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise OccupancyV3NotReadyError("Enhanced model metadata is unreadable") from exc
        if metadata.get("schema_version") != OCCUPANCY_SCHEMA_VERSION:
            raise OccupancyV3NotReadyError("Enhanced model schema is unsupported")
        if metadata.get("onnx_sha256") != sha256_file(model_path):
            raise OccupancyV3NotReadyError("Enhanced model checksum is invalid")
        if not metadata.get("frozen") or "final_holdout" not in metadata:
            raise OccupancyV3NotReadyError("Enhanced model has not passed the frozen holdout gate")
        providers = ["CPUExecutionProvider"]
        return cls(
            session=ort.InferenceSession(str(model_path), providers=providers),
            threshold=float(metadata["decision_threshold"]),
            temperature=max(float(metadata.get("temperature", 1.0)), 1e-3),
            metadata=metadata,
        )

    def probabilities(self, images: list[Image.Image]) -> tuple[np.ndarray, float]:
        if not images:
            return np.empty(0, dtype=np.float32), 0.0
        transform = preprocessing(False)
        batch = np.stack([transform(image.convert("RGB")).numpy() for image in images])
        started = perf_counter()
        logits = self.session.run(["logit"], {"image": batch.astype(np.float32)})[0]
        elapsed = (perf_counter() - started) * 1_000
        logits = np.asarray(logits, dtype=np.float64).reshape(-1) / self.temperature
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits, -40, 40)))
        return probabilities.astype(np.float32), elapsed

    def predict(self, images: list[Image.Image]) -> tuple[list[dict[str, object]], float]:
        probabilities, elapsed = self.probabilities(images)
        results = []
        for probability in probabilities:
            occupied = bool(float(probability) >= self.threshold)
            results.append(
                {
                    "predicted_occupied": occupied,
                    "occupied_probability": round(float(probability), 6),
                    "confidence": round(float(max(probability, 1.0 - probability)), 6),
                }
            )
        return results, elapsed


def occupancy_v3_status(model_root: Path) -> dict[str, object]:
    try:
        predictor = OccupancyV3Predictor.load(model_root)
    except OccupancyV3NotReadyError as exc:
        return {"ready": False, "model_name": OCCUPANCY_MODEL_NAME, "reason": str(exc)}
    return {
        "ready": True,
        "model_name": OCCUPANCY_MODEL_NAME,
        "architecture": predictor.metadata.get("architecture"),
        "decision_threshold": predictor.threshold,
        "onnx_sha256": predictor.metadata.get("onnx_sha256"),
        "final_holdout": predictor.metadata.get("final_holdout"),
        "calibration": predictor.metadata.get("calibration"),
    }
