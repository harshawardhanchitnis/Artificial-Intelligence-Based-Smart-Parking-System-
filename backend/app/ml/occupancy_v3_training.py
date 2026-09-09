"""Model definitions and ONNX export for the enhanced occupancy classifier.

Split out of ``app.ml.occupancy_v3`` so that the serving path can load and run
the exported model without PyTorch present in the process.  Nothing the FastAPI
application imports reaches this module; ``app.ml.training_v3`` and the
``train-occupancy-v3`` CLI do, and they are the only callers that need torch.

The architectures and the export call are unchanged -- this file is where they
already were, moved.  Re-exporting a model from these definitions produces the
same ONNX graph the installed artifact was built from.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

from app.ml.occupancy_v3 import (
    IMAGE_SIZE,
    OCCUPANCY_METADATA,
    OCCUPANCY_MODEL_NAME,
    OCCUPANCY_ONNX,
    OCCUPANCY_SCHEMA_VERSION,
    atomic_json,
    sha256_file,
)


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
    """Training-time transform.

    Inference uses ``app.ml.preprocessing.imagenet_chw``, which reproduces the
    ``training=False`` branch of this pipeline exactly without torchvision.
    """
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
