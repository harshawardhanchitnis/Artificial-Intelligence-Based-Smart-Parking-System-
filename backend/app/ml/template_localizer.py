"""Recognise a registered fixed camera and reuse the layout it already has.

Two mechanisms live here: an ONNX layout classifier that names the camera, and
an ORB/RANSAC template registry that re-projects a stored layout onto a fresh
frame.  Both run on every image request, so this module carries no training
framework -- the classifier's architecture and its ONNX export are in
``app.ml.template_localizer_training``.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image

from app.datasets.integrity import atomic_json, deterministic_score, sha256_file
from app.ml import registry
from app.ml.geometry import order_polygon, validate_polygon
from app.ml.preprocessing import imagenet_chw
from app.ml.registry import inference_providers, session_options

TEMPLATE_DIRECTORY = "parking-layout-templates"
TEMPLATE_MANIFEST = "parking-layout-templates.json"
LAYOUT_CLASSIFIER_ONNX = "parking-layout-classifier-v1.onnx"
LAYOUT_CLASSIFIER_METADATA = "parking-layout-classifier-v1.json"
LAYOUT_INPUT_SIZE = 224
WIDE_ASPECT_THRESHOLD = 1.55


def layout_key(row: dict[str, object]) -> str:
    source_id = str(row["source_id"])
    return f"{row['dataset']}:{source_id.split('/', 1)[0]}"


def constrain_layout_logits(
    logits: np.ndarray, labels: list[str], *, width: int, height: int
) -> np.ndarray:
    """Restrict fixed-camera classification to the matching source aspect family.

    PKLot cameras are 16:9 while CNRPark+EXT cameras are 4:3.  The ratio is an
    observable camera property (and survives ordinary resizing), so using it
    prevents low-light imagery from being confused across incompatible camera
    families without looking at occupancy labels or protected partitions.
    """
    aspect_ratio = width / max(height, 1)
    expected_dataset = "PKLot" if aspect_ratio >= WIDE_ASPECT_THRESHOLD else "CNRPark+EXT"
    constrained = np.asarray(logits, dtype=np.float32).copy()
    for index, label in enumerate(labels):
        if not label.startswith(f"{expected_dataset}:"):
            constrained[index] = -1e9
    return constrained


def select_template_rows(
    rows: list[dict[str, object]], *, per_layout: int = 3
) -> list[dict[str, object]]:
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(layout_key(row), []).append(row)
    selected = []
    for _key, candidates in sorted(grouped.items()):
        conditions: dict[str, list[dict[str, object]]] = {}
        for row in candidates:
            conditions.setdefault(str(row.get("condition", "unknown")).lower(), []).append(row)
        chosen: list[dict[str, object]] = []
        for condition in sorted(conditions):
            chosen.append(
                min(
                    conditions[condition],
                    key=lambda row: deterministic_score(str(row["source_id"])),
                )
            )
        remaining = sorted(
            (row for row in candidates if row not in chosen),
            key=lambda row: deterministic_score(str(row["source_id"])),
        )
        selected.extend((chosen + remaining)[:per_layout])
    return selected


def _features(image: Image.Image):
    rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    scale = min(1.0, 960.0 / max(width, 1))
    if scale < 1.0:
        rgb = cv2.resize(rgb, (round(width * scale), round(height * scale)))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    detector = cv2.ORB_create(nfeatures=3500, fastThreshold=10)
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    points = np.asarray([keypoint.pt for keypoint in keypoints], dtype=np.float32)
    return points, descriptors, (rgb.shape[1], rgb.shape[0])


@dataclass
class LayoutTemplate:
    layout_key: str
    source_id: str
    image_path: Path
    slots: list[dict[str, object]]
    points: np.ndarray
    descriptors: np.ndarray | None
    size: tuple[int, int]


class TemplateLayoutRegistry:
    def __init__(self, templates: list[LayoutTemplate]) -> None:
        self.templates = templates

    @classmethod
    def from_rows(
        cls, root: Path, rows: list[dict[str, object]], *, per_layout: int = 3
    ) -> TemplateLayoutRegistry:
        templates = []
        for row in select_template_rows(rows, per_layout=per_layout):
            image_path = root / str(row["image_path"])
            with Image.open(image_path) as image:
                points, descriptors, size = _features(image)
            templates.append(
                LayoutTemplate(
                    layout_key(row),
                    str(row["source_id"]),
                    image_path,
                    row["slots"],  # type: ignore[arg-type]
                    points,
                    descriptors,
                    size,
                )
            )
        return cls(templates)

    @classmethod
    def load(cls, model_root: Path) -> TemplateLayoutRegistry:
        manifest_path = model_root / TEMPLATE_MANIFEST
        if not manifest_path.is_file():
            return cls([])
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        templates = []
        for row in payload["templates"]:
            image_path = model_root / str(row["image_path"])
            if sha256_file(image_path) != row["image_sha256"]:
                raise RuntimeError(f"Template checksum mismatch: {image_path.name}")
            with Image.open(image_path) as image:
                points, descriptors, size = _features(image)
            templates.append(
                LayoutTemplate(
                    str(row["layout_key"]),
                    str(row["source_id"]),
                    image_path,
                    row["slots"],
                    points,
                    descriptors,
                    size,
                )
            )
        return cls(templates)

    def detect(
        self, image: Image.Image, *, expected_layout: str | None = None
    ) -> dict[str, object]:
        query_points, query_descriptors, query_size = _features(image)
        if query_descriptors is None or len(query_points) < 12:
            return {"status": "unsupported_layout", "confidence": 0.0, "slots": []}
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        matches = []
        for template in self.templates:
            if expected_layout is not None and template.layout_key != expected_layout:
                continue
            if template.descriptors is None or len(template.points) < 12:
                continue
            pairs = matcher.knnMatch(template.descriptors, query_descriptors, k=2)
            good = [first for first, second in pairs if first.distance < 0.8 * second.distance]
            if len(good) < 8:
                continue
            source = np.asarray([template.points[item.queryIdx] for item in good])
            destination = np.asarray([query_points[item.trainIdx] for item in good])
            matrix, mask = cv2.findHomography(source, destination, cv2.RANSAC, 4.0)
            if matrix is None or mask is None:
                continue
            inliers = int(mask.sum())
            ratio = inliers / len(good)
            score = inliers * ratio
            matches.append((score, inliers, ratio, template, matrix))
        if not matches:
            return {"status": "unsupported_layout", "confidence": 0.0, "slots": []}
        matches.sort(key=lambda item: item[0], reverse=True)
        score, inliers, ratio, template, matrix = matches[0]
        competing = max(
            (item[0] for item in matches[1:] if item[3].layout_key != template.layout_key),
            default=0.0,
        )
        margin = max(0.0, 1.0 - competing / max(score, 1e-6))
        confidence = min(1.0, inliers / 80.0) * ratio * (0.6 + 0.4 * margin)
        if inliers < 8 or ratio < 0.2 or confidence < 0.05:
            return {"status": "unsupported_layout", "confidence": confidence, "slots": []}
        template_width, template_height = template.size
        query_width, query_height = query_size
        slots = []
        for index, slot in enumerate(template.slots, 1):
            points = np.asarray(slot["polygon"], dtype=np.float32)
            points[:, 0] *= template_width - 1
            points[:, 1] *= template_height - 1
            mapped = cv2.perspectiveTransform(points[None, :, :], matrix)[0]
            mapped[:, 0] /= query_width - 1
            mapped[:, 1] /= query_height - 1
            polygon = order_polygon(mapped.clip(0.0, 1.0).tolist())
            if validate_polygon(polygon).valid:
                slots.append(
                    {
                        "id": str(index),
                        "polygon": polygon,
                        "localization_confidence": float(confidence),
                        "corner_confidence": float(confidence),
                    }
                )
        status = "success" if confidence >= 0.55 else "success_with_warnings"
        return {
            "status": status,
            "confidence": round(float(confidence), 6),
            "slots": slots,
            "layout_key": template.layout_key,
            "reference_source_id": template.source_id,
            "inlier_count": inliers,
            "inlier_ratio": round(float(ratio), 6),
            "method": "training-template-orb-ransac",
        }


def export_template_registry(
    registry: TemplateLayoutRegistry, model_root: Path
) -> dict[str, object]:
    destination = model_root / TEMPLATE_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for index, template in enumerate(registry.templates, 1):
        suffix = template.image_path.suffix.lower() or ".jpg"
        target = destination / f"template-{index:02d}{suffix}"
        shutil.copyfile(template.image_path, target)
        records.append(
            {
                "layout_key": template.layout_key,
                "source_id": template.source_id,
                "image_path": target.relative_to(model_root).as_posix(),
                "image_sha256": sha256_file(target),
                "slots": template.slots,
            }
        )
    manifest = {
        "schema_version": "1.0",
        "method": "fixed-camera training-template ORB/RANSAC registration",
        "template_count": len(records),
        "layouts": sorted({str(row["layout_key"]) for row in records}),
        "templates": records,
    }
    atomic_json(model_root / TEMPLATE_MANIFEST, manifest)
    return {
        "template_manifest": TEMPLATE_MANIFEST,
        "template_manifest_sha256": sha256_file(model_root / TEMPLATE_MANIFEST),
        "template_count": len(records),
        "layout_count": len(manifest["layouts"]),
    }


@dataclass
class LayoutClassifierPredictor:
    session: ort.InferenceSession
    labels: list[str]
    canonical_slots: dict[str, list[dict[str, object]]]
    confidence_threshold: float

    @classmethod
    def load(cls, model_root: Path) -> LayoutClassifierPredictor:
        """Return the shared classifier, building it once per model file version."""
        metadata_path = model_root / LAYOUT_CLASSIFIER_METADATA
        if not metadata_path.is_file():
            raise RuntimeError("Fixed-camera layout classifier is not installed")
        metadata = registry.read_metadata(metadata_path)
        model_path = model_root / str(metadata["onnx_file"])
        signature = registry.file_signature(model_path, metadata_path)

        def build() -> LayoutClassifierPredictor:
            digest = registry.cached(
                f"sha256::{model_path}",
                registry.file_signature(model_path),
                lambda: sha256_file(model_path),
            )
            if not model_path.is_file() or digest != metadata["onnx_sha256"]:
                raise RuntimeError("Fixed-camera layout classifier checksum is invalid")
            return cls(
                ort.InferenceSession(
                    str(model_path),
                    sess_options=session_options(),
                    providers=inference_providers(),
                ),
                [str(value) for value in metadata["labels"]],
                metadata["canonical_slots"],
                float(metadata["confidence_threshold"]),
            )

        return registry.cached("layout_classifier", signature, build)

    def detect(self, image: Image.Image) -> dict[str, object]:
        rgb_image = image.convert("RGB")
        grayscale = cv2.cvtColor(np.asarray(rgb_image), cv2.COLOR_RGB2GRAY)
        contrast = float(grayscale.std())
        # This floor sits below every train/validation parking image (minimum
        # observed training contrast 11.03; validation 17.10) and rejects blank
        # or diagram-like uploads before a closed-set classifier can overclaim.
        if contrast < 10.0:
            return {
                "status": "unsupported_layout",
                "confidence": 0.0,
                "slots": [],
                "method": "photographic-content-preflight",
                "reason": (
                    "Image lacks enough photographic contrast for reliable layout recognition"
                ),
            }
        inputs = imagenet_chw(rgb_image, LAYOUT_INPUT_SIZE)[None, :, :, :]
        logits = self.session.run(["layout_logits"], {"image": inputs})[0][0]
        logits = constrain_layout_logits(
            logits, self.labels, width=image.width, height=image.height
        )
        probabilities = np.exp(logits - np.max(logits))
        probabilities /= probabilities.sum()
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        label = self.labels[index]
        if confidence < self.confidence_threshold:
            return {
                "status": "unsupported_layout",
                "confidence": round(confidence, 6),
                "slots": [],
                "method": "fixed-camera-layout-classifier",
            }
        slots = [
            {
                "id": str(slot_index),
                "polygon": slot["polygon"],
                "localization_confidence": confidence,
                "corner_confidence": confidence,
            }
            for slot_index, slot in enumerate(self.canonical_slots[label], 1)
        ]
        return {
            "status": "success" if confidence >= 0.75 else "success_with_warnings",
            "confidence": round(confidence, 6),
            "slots": slots,
            "layout_key": label,
            "method": "fixed-camera-layout-classifier",
        }
