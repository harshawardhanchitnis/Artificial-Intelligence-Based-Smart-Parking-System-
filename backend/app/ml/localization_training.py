"""Model definition, targets and ONNX export for the V1 slot localizer.

Split out of ``app.ml.localization`` so the serving path can stay free of the
training framework.  ``app.ml.localizer_training`` and the
``train-slot-localizer`` CLI import this; the FastAPI application does not.

The network, the target encoding and the export call are unchanged -- moved,
not rewritten -- so retraining and re-exporting reproduce the installed
artifact.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

from app.ml.geometry import order_polygon
from app.ml.localization import (
    INPUT_HEIGHT,
    INPUT_WIDTH,
    LOCALIZER_METADATA,
    LOCALIZER_NAME,
    LOCALIZER_ONNX,
    LOCALIZER_SCHEMA,
    MAX_CORNER_OFFSET,
    sha256_file,
)


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
        "onnx_sha256": sha256_file(destination),
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
