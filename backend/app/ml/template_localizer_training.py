"""Model definition and ONNX export for the fixed-camera layout classifier.

Split out of ``app.ml.template_localizer`` for the same reason as the occupancy
split: the serving path runs the exported ONNX graph and must not carry the
framework that produced it.  ``app.ml.localizer_training`` and the
``train-slot-localizer`` CLI import this; the FastAPI application does not.

The architecture and the export call are unchanged -- moved, not rewritten.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms

from app.datasets.integrity import atomic_json, sha256_file
from app.ml.template_localizer import (
    LAYOUT_CLASSIFIER_METADATA,
    LAYOUT_CLASSIFIER_ONNX,
    LAYOUT_INPUT_SIZE,
    WIDE_ASPECT_THRESHOLD,
)


def layout_preprocessing(training: bool = False) -> transforms.Compose:
    """Training-time transform.

    Inference uses ``app.ml.preprocessing.imagenet_chw``, which reproduces the
    ``training=False`` branch of this pipeline exactly without torchvision.
    """
    operations: list[object] = [transforms.Resize((LAYOUT_INPUT_SIZE, LAYOUT_INPUT_SIZE))]
    if training:
        operations.extend(
            [
                transforms.ColorJitter(brightness=0.35, contrast=0.35, saturation=0.2),
                transforms.RandomAutocontrast(p=0.2),
            ]
        )
    operations.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    return transforms.Compose(operations)  # type: ignore[arg-type]


def build_layout_classifier(class_count: int, *, pretrained: bool = False) -> nn.Module:
    model = models.mobilenet_v3_small(weights="DEFAULT" if pretrained else None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, class_count)
    return model


def export_layout_classifier(
    model: nn.Module,
    model_root: Path,
    *,
    labels: list[str],
    canonical_slots: dict[str, list[dict[str, object]]],
    confidence_threshold: float,
) -> dict[str, object]:
    destination = model_root / LAYOUT_CLASSIFIER_ONNX
    temporary = destination.with_suffix(".onnx.tmp")
    model.eval().cpu()
    torch.onnx.export(
        model,
        torch.zeros(1, 3, LAYOUT_INPUT_SIZE, LAYOUT_INPUT_SIZE),
        temporary,
        input_names=["image"],
        output_names=["layout_logits"],
        dynamic_axes={"image": {0: "batch"}, "layout_logits": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
    temporary.replace(destination)
    metadata = {
        "schema_version": "1.0",
        "labels": labels,
        "canonical_slots": canonical_slots,
        "confidence_threshold": confidence_threshold,
        "onnx_file": LAYOUT_CLASSIFIER_ONNX,
        "onnx_sha256": sha256_file(destination),
        "scope": "registered fixed-camera layouts; unfamiliar viewpoints are rejected",
        "layout_family_constraint": {
            "method": "source aspect-ratio family",
            "wide_aspect_threshold": WIDE_ASPECT_THRESHOLD,
            "wide_family": "PKLot",
            "standard_family": "CNRPark+EXT",
        },
    }
    atomic_json(model_root / LAYOUT_CLASSIFIER_METADATA, metadata)
    return {
        "layout_classifier_file": LAYOUT_CLASSIFIER_ONNX,
        "layout_classifier_sha256": metadata["onnx_sha256"],
        "layout_classifier_metadata": LAYOUT_CLASSIFIER_METADATA,
    }
