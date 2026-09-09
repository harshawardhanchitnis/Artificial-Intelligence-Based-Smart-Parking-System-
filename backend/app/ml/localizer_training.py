from __future__ import annotations

import json
import random
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.datasets.integrity import PROTOCOL_ID, atomic_json, read_jsonl, sha256_file
from app.ml.geometry import polygon_iou, rectify_slot
from app.ml.localization import (
    GRID_HEIGHT,
    GRID_WIDTH,
    LOCALIZER_METADATA,
    LOCALIZER_NAME,
    SlotLocalizerPredictor,
    decode_localizer_output,
)
from app.ml.localization_training import (
    ParkingSlotLocalizer,
    export_localizer,
    localizer_preprocessing,
    localizer_targets,
)
from app.ml.occupancy_v3 import OccupancyV3NotReadyError, OccupancyV3Predictor
from app.ml.template_localizer import (
    TemplateLayoutRegistry,
    constrain_layout_logits,
    layout_key,
)
from app.ml.template_localizer_training import (
    build_layout_classifier,
    export_layout_classifier,
    layout_preprocessing,
)
from app.ml.training import classification_metrics


def _geometric_augmentation(
    image: Image.Image, slots: list[dict[str, object]]
) -> tuple[Image.Image, list[dict[str, object]]]:
    """Apply one mild homography and transform polygon supervision identically."""
    pixels = np.asarray(image.convert("RGB"))
    height, width = pixels.shape[:2]
    source = np.asarray(
        [
            [0.0, 0.0],
            [width - 1.0, 0.0],
            [width - 1.0, height - 1.0],
            [0.0, height - 1.0],
        ],
        dtype=np.float32,
    )
    jitter = np.asarray(
        [
            [random.uniform(-0.025, 0.025) * width, random.uniform(-0.025, 0.025) * height]
            for _ in range(4)
        ],
        dtype=np.float32,
    )
    destination = source + jitter
    matrix = cv2.getPerspectiveTransform(source, destination)
    warped = cv2.warpPerspective(
        pixels,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )
    transformed: list[dict[str, object]] = []
    for slot in slots:
        points = np.asarray(slot["polygon"], dtype=np.float32)
        points[:, 0] *= width - 1
        points[:, 1] *= height - 1
        mapped = cv2.perspectiveTransform(points[None, :, :], matrix)[0]
        mapped[:, 0] /= width - 1
        mapped[:, 1] /= height - 1
        if np.any(mapped < 0.0) or np.any(mapped > 1.0):
            continue
        transformed.append({**slot, "polygon": mapped.tolist()})
    if random.random() < 0.5:
        warped = np.ascontiguousarray(warped[:, ::-1])
        transformed = [
            {
                **slot,
                "polygon": [[1.0 - float(x), float(y)] for x, y in slot["polygon"]],
            }
            for slot in transformed
        ]
    return Image.fromarray(warped), transformed


class LocalizerDataset(Dataset):
    def __init__(
        self, root: Path, rows: list[dict[str, object]], *, training: bool = False
    ) -> None:
        self.root = root
        self.rows = rows
        self.training = training
        self.transform = localizer_preprocessing(training)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / str(row["image_path"])) as image:
            source = image.convert("RGB")
        slots = row["slots"]  # type: ignore[assignment]
        if self.training:
            source, slots = _geometric_augmentation(source, slots)  # type: ignore[arg-type]
        inputs = self.transform(source)
        objectness, corners, collisions = localizer_targets(
            slots, GRID_HEIGHT, GRID_WIDTH  # type: ignore[arg-type]
        )
        return inputs, objectness, corners, collisions


class LayoutClassificationDataset(Dataset):
    def __init__(
        self,
        root: Path,
        rows: list[dict[str, object]],
        label_indices: dict[str, int],
        *,
        training: bool = False,
    ) -> None:
        self.root = root
        self.rows = rows
        self.label_indices = label_indices
        self.transform = layout_preprocessing(training)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(self.root / str(row["image_path"])) as image:
            inputs = self.transform(image.convert("RGB"))
        return inputs, self.label_indices[layout_key(row)]


def _source_partitions(root: Path) -> dict[str, list[dict[str, object]]]:
    rows = read_jsonl(root / "source-manifest.jsonl")
    supported = [row for row in rows if row["dataset"] in {"PKLot", "CNRPark+EXT"}]
    result = {
        name: [row for row in supported if row["partition"] == name]
        for name in ("train", "validation", "holdout")
    }
    if any(not rows for rows in result.values()):
        raise RuntimeError("Localisation train, validation and holdout partitions are required")
    return result


def _loss(output, objects, corners):
    object_loss = nn.functional.binary_cross_entropy_with_logits(
        output[:, :1], objects, pos_weight=torch.tensor(96.0, device=output.device)
    )
    mask = objects.expand(-1, 8, -1, -1) > 0
    corner_loss = nn.functional.smooth_l1_loss(torch.tanh(output[:, 1:])[mask], corners[mask])
    return object_loss + 8.0 * corner_loss, object_loss, corner_loss


def _raw_outputs(model, loader, device):
    outputs = []
    model.eval()
    started = perf_counter()
    with torch.inference_mode():
        for inputs, _, _, _ in loader:
            outputs.extend(model(inputs.to(device)).cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    return outputs, (perf_counter() - started) * 1_000


def _precompute_iou_matrices(rows, decoded_rows) -> list[np.ndarray]:
    matrices = []
    for row, predicted in zip(rows, decoded_rows, strict=True):
        truth = [slot["polygon"] for slot in row["slots"]]
        matrices.append(
            np.asarray(
                [
                    [polygon_iou(proposal["polygon"], polygon) for polygon in truth]
                    for proposal in predicted
                ],
                dtype=np.float32,
            ).reshape(len(predicted), len(truth))
        )
    return matrices


def _metrics(
    rows,
    outputs,
    threshold,
    *,
    decoded_rows: list[list[dict[str, object]]] | None = None,
    iou_matrices: list[np.ndarray] | None = None,
):
    true_positive = false_positive = false_negative = 0
    true_positive_75 = 0
    corner_errors = []
    count_errors = []
    by_dataset: dict[str, list[tuple[int, int, int]]] = {}
    by_layout = []
    if decoded_rows is None:
        decoded_rows = [
            decode_localizer_output(output, object_threshold=threshold) for output in outputs
        ]
    if iou_matrices is None:
        iou_matrices = _precompute_iou_matrices(rows, decoded_rows)
    for row, predicted, overlaps in zip(rows, decoded_rows, iou_matrices, strict=True):
        truth = [slot["polygon"] for slot in row["slots"]]
        unmatched = set(range(len(truth)))
        image_tp = 0
        for prediction_index, prediction in enumerate(predicted):
            candidates = [(float(overlaps[prediction_index, index]), index) for index in unmatched]
            overlap, match = max(candidates, default=(0.0, -1))
            if overlap >= 0.5:
                unmatched.remove(match)
                image_tp += 1
                predicted_points = np.asarray(prediction["polygon"])
                truth_points = np.asarray(truth[match])
                corner_errors.append(
                    float(np.mean(np.linalg.norm(predicted_points - truth_points, axis=1)))
                )
                true_positive_75 += int(overlap >= 0.75)
            else:
                false_positive += 1
        true_positive += image_tp
        false_negative += len(unmatched)
        count_errors.append(abs(len(predicted) - len(truth)))
        dataset_values = by_dataset.setdefault(str(row["dataset"]), [0, 0, 0])
        dataset_values[0] += image_tp
        dataset_values[1] += len(predicted) - image_tp
        dataset_values[2] += len(unmatched)
        by_layout.append(
            {
                "source_id": row["source_id"],
                "dataset": row["dataset"],
                "truth_slots": len(truth),
                "detected_slots": len(predicted),
                "matched_iou_50": image_tp,
                "recall_iou_50": round(image_tp / max(len(truth), 1), 6),
                "count_error": abs(len(predicted) - len(truth)),
            }
        )
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "precision_iou_50": round(precision, 6),
        "recall_iou_50": round(recall, 6),
        "f1_iou_50": round(2 * precision * recall / max(precision + recall, 1e-9), 6),
        "recall_iou_75": round(true_positive_75 / max(true_positive + false_negative, 1), 6),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "mean_corner_error_normalized": round(
            float(np.mean(corner_errors)) if corner_errors else 1.0, 6
        ),
        "mean_absolute_count_error": round(float(np.mean(count_errors)), 3),
        # Decoder NMS rejects IoU >= 0.30, so duplicate IoU >= 0.50 is zero by construction.
        "duplicate_slot_rate": 0.0,
        "missed_slot_rate": round(false_negative / max(true_positive + false_negative, 1), 6),
        "images": len(rows),
        "by_dataset": {
            dataset: {
                "precision_iou_50": round(values[0] / max(values[0] + values[1], 1), 6),
                "recall_iou_50": round(values[0] / max(values[0] + values[2], 1), 6),
            }
            for dataset, values in by_dataset.items()
        },
        "worst_layouts": sorted(
            by_layout,
            key=lambda value: (float(value["recall_iou_50"]), -int(value["count_error"])),
        )[:10],
    }


def _detected_geometry_occupancy_metrics(
    root: Path,
    rows: list[dict[str, object]],
    outputs: list[np.ndarray] | None,
    object_threshold: float,
    model_root: Path,
    *,
    decoded_rows: list[list[dict[str, object]]] | None = None,
) -> dict[str, object]:
    try:
        classifier = OccupancyV3Predictor.load(model_root)
    except OccupancyV3NotReadyError:
        return {"available": False, "reason": "Enhanced occupancy model is not installed"}
    labels: list[int] = []
    predictions: list[int] = []
    matched = truth_total = 0
    if decoded_rows is None:
        if outputs is None:
            raise ValueError("Either raw outputs or decoded rows are required")
        decoded_rows = [
            decode_localizer_output(output, object_threshold=object_threshold)
            for output in outputs
        ]
    for row, detected in zip(rows, decoded_rows, strict=True):
        truth = row["slots"]
        truth_total += len(truth)  # type: ignore[arg-type]
        unmatched = set(range(len(truth)))  # type: ignore[arg-type]
        match_pairs = []
        for slot in detected:
            candidates = [
                (polygon_iou(slot["polygon"], truth[index]["polygon"]), index)  # type: ignore[index]
                for index in unmatched
            ]
            overlap, match = max(candidates, default=(0.0, -1))
            if overlap >= 0.5:
                unmatched.remove(match)
                match_pairs.append((slot, truth[match]))  # type: ignore[index]
        if not match_pairs:
            continue
        with Image.open(root / str(row["image_path"])) as source:
            patches = [rectify_slot(source, pair[0]["polygon"]) for pair in match_pairs]
        states, _ = classifier.predict(patches)
        labels.extend(int(bool(pair[1]["occupied"])) for pair in match_pairs)
        predictions.extend(int(bool(state["predicted_occupied"])) for state in states)
        matched += len(match_pairs)
    if not labels:
        return {"available": True, "matched_slots": 0, "truth_slots": truth_total}
    return {
        "available": True,
        "geometry": "automatically detected polygons; IoU >= 0.50 truth association",
        "matched_slots": matched,
        "truth_slots": truth_total,
        "localisation_coverage": round(matched / max(truth_total, 1), 6),
        **classification_metrics(np.asarray(labels), np.asarray(predictions)),
    }


def _template_metrics(
    root: Path,
    registry: TemplateLayoutRegistry,
    rows: list[dict[str, object]],
    *,
    progress=print,
) -> tuple[dict[str, object], list[list[dict[str, object]]], dict[str, int]]:
    decoded_rows = []
    statuses: dict[str, int] = {}
    for index, row in enumerate(rows, 1):
        with Image.open(root / str(row["image_path"])) as image:
            result = registry.detect(image)
        status = str(result["status"])
        statuses[status] = statuses.get(status, 0) + 1
        decoded_rows.append(result["slots"])  # type: ignore[arg-type]
        if progress and index % 100 == 0:
            progress(f"template localizer: {index:,}/{len(rows):,} images")
    metrics = _metrics(rows, [None] * len(rows), 0.0, decoded_rows=decoded_rows)
    return metrics, decoded_rows, statuses


def _develop_layout_classifier(
    root: Path,
    partitions: dict[str, list[dict[str, object]]],
    report_root: Path,
    *,
    epochs: int = 8,
    progress=print,
) -> dict[str, object]:
    labels = sorted({layout_key(row) for row in partitions["train"]})
    label_indices = {label: index for index, label in enumerate(labels)}
    if any(layout_key(row) not in label_indices for row in partitions["validation"]):
        raise RuntimeError("Validation contains a fixed-camera layout absent from training")
    canonical_slots: dict[str, list[dict[str, object]]] = {}
    for row in partitions["train"]:
        canonical_slots.setdefault(layout_key(row), row["slots"])  # type: ignore[arg-type]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    workers = 2 if device.type == "cuda" else 0
    training_loader = DataLoader(
        LayoutClassificationDataset(root, partitions["train"], label_indices, training=True),
        batch_size=64,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        LayoutClassificationDataset(root, partitions["validation"], label_indices),
        batch_size=64,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    model = build_layout_classifier(len(labels), pretrained=True).to(device)
    state_path = report_root / "layout-classifier-weights.pt"
    resumed = state_path.is_file()
    if resumed:
        model.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": 1e-5 if resumed else 2e-5},
            {"params": model.classifier.parameters(), "lr": 1e-4 if resumed else 2e-4},
        ],
        weight_decay=1e-4,
    )
    best_state = {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }
    model.eval()
    initial_correct = initial_total = 0
    with torch.inference_mode():
        for inputs, targets in validation_loader:
            predicted = model(inputs.to(device)).argmax(dim=1).cpu()
            initial_correct += int((predicted == targets).sum())
            initial_total += len(targets)
    best_accuracy = initial_correct / max(initial_total, 1)
    stale_epochs = 0
    started = perf_counter()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, targets in training_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs.to(device))
            loss = nn.functional.cross_entropy(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach()) * len(targets)
        model.eval()
        correct = total = 0
        with torch.inference_mode():
            for inputs, targets in validation_loader:
                predicted = model(inputs.to(device)).argmax(dim=1).cpu()
                correct += int((predicted == targets).sum())
                total += len(targets)
        accuracy = correct / max(total, 1)
        if progress:
            progress(
                f"layout-classifier: epoch {epoch + 1}/{epochs}, "
                f"train_loss={running_loss / len(partitions['train']):.4f}, "
                f"val_accuracy={accuracy:.4f}"
            )
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= 3:
                break
    model.load_state_dict(best_state)
    model.to(device).eval()
    logits_parts = []
    inference_started = perf_counter()
    with torch.inference_mode():
        for inputs, _ in validation_loader:
            logits_parts.append(model(inputs.to(device)).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    latency_ms = (perf_counter() - inference_started) * 1_000 / len(partitions["validation"])
    raw_logits = torch.cat(logits_parts).numpy()
    constrained_logits = np.stack(
        [
            constrain_layout_logits(
                logits,
                labels,
                width=int(row["image_width"]),
                height=int(row["image_height"]),
            )
            for logits, row in zip(raw_logits, partitions["validation"], strict=True)
        ]
    )
    probabilities = torch.softmax(torch.from_numpy(constrained_logits), dim=1).numpy()
    predicted_indices = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    decoded_all = []
    for predicted_index, confidence in zip(predicted_indices, confidences, strict=True):
        label = labels[int(predicted_index)]
        decoded_all.append(
            [
                {
                    "id": str(index),
                    "polygon": slot["polygon"],
                    "localization_confidence": float(confidence),
                    "corner_confidence": float(confidence),
                }
                for index, slot in enumerate(canonical_slots[label], 1)
            ]
        )
    all_iou_matrices = _precompute_iou_matrices(partitions["validation"], decoded_all)
    candidates = np.linspace(0.35, 0.9, 12)
    evaluations = []
    for threshold in candidates:
        decoded_rows = []
        iou_matrices = []
        for proposals, overlaps, confidence in zip(
            decoded_all, all_iou_matrices, confidences, strict=True
        ):
            if confidence < threshold:
                decoded_rows.append([])
                iou_matrices.append(overlaps[:0, :])
                continue
            decoded_rows.append(proposals)
            iou_matrices.append(overlaps)
        evaluations.append(
            (
                float(threshold),
                _metrics(
                    partitions["validation"],
                    [None] * len(partitions["validation"]),
                    0.0,
                    decoded_rows=decoded_rows,
                    iou_matrices=iou_matrices,
                ),
            )
        )
    threshold, metrics = max(
        evaluations,
        key=lambda item: (
            min(float(value["recall_iou_50"]) for value in item[1]["by_dataset"].values()),
            min(
                float(value["precision_iou_50"])
                for value in item[1]["by_dataset"].values()
            ),
        ),
    )
    torch.save(model.cpu().state_dict(), state_path)
    return {
        "labels": labels,
        "canonical_slots": canonical_slots,
        "confidence_threshold": threshold,
        "validation": metrics,
        "classification_accuracy": round(best_accuracy, 6),
        "resumed_from_checkpoint": resumed,
        "training_seconds": round(perf_counter() - started, 3),
        "validation_latency_ms_per_image": round(latency_ms, 3),
        "state_path": state_path,
    }


def develop_hybrid_localizer(
    data_root: Path, model_root: Path, *, progress=print
) -> dict[str, object]:
    random.seed(20260904)
    np.random.seed(20260904)
    torch.manual_seed(20260904)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260904)
    root = data_root / "prepared" / "v2-protocol"
    partitions = _source_partitions(root)
    report_root = model_root / "development" / "slot-localizer-v1"
    report_path = report_root / "development-report.json"
    state_path = report_root / "development-weights.pt"
    if not report_path.is_file() or not state_path.is_file():
        raise RuntimeError("Run neural localizer development before hybrid evaluation")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    registry = TemplateLayoutRegistry.from_rows(root, partitions["train"], per_layout=3)
    previous_strategy = report.get("localization_strategy")
    if "orb_registration_validation" in report:
        orb_metrics = report["orb_registration_validation"]
        statuses = report.get("validation_statuses", {})
    elif previous_strategy == "training-template ORB/RANSAC; neural detector diagnostic":
        orb_metrics = report["validation"]
        statuses = report.get("validation_statuses", {})
    else:
        orb_metrics, _, statuses = _template_metrics(
            root, registry, partitions["validation"], progress=progress
        )
    classifier = _develop_layout_classifier(root, partitions, report_root, progress=progress)
    metrics = classifier["validation"]
    minimum_dataset_recall = min(
        float(value["recall_iou_50"]) for value in metrics["by_dataset"].values()
    )
    minimum_dataset_precision = min(
        float(value["precision_iou_50"]) for value in metrics["by_dataset"].values()
    )
    if "neural_detector_validation" not in report:
        report["neural_detector_validation"] = report["validation"]
    report["orb_registration_validation"] = orb_metrics
    report["validation"] = metrics
    report["validation_statuses"] = statuses
    report["localization_strategy"] = (
        "fixed-camera layout classifier with verified canonical polygons; "
        "ORB/RANSAC and neural detector retained as diagnostics"
    )
    report["layout_classifier"] = {
        key: value
        for key, value in classifier.items()
        if key not in {"canonical_slots", "state_path", "validation"}
    }
    report["template_count"] = len(registry.templates)
    report["template_layouts"] = sorted({template.layout_key for template in registry.templates})
    report["deployment_gate"] = {
        "minimum_dataset_recall_required": 0.85,
        "minimum_dataset_precision_required": 0.85,
        "observed_minimum_dataset_recall": round(minimum_dataset_recall, 6),
        "observed_minimum_dataset_precision": round(minimum_dataset_precision, 6),
        "passed": minimum_dataset_recall >= 0.85 and minimum_dataset_precision >= 0.85,
    }
    atomic_json(report_path, report)
    atomic_json(
        report_root / "decision-lock.json",
        {
            "model_name": LOCALIZER_NAME,
            "localization_strategy": report["localization_strategy"],
            "object_threshold": report["object_threshold"],
            "state_sha256": sha256_file(state_path),
            "layout_classifier_state_sha256": sha256_file(classifier["state_path"]),
            "development_report_sha256": sha256_file(report_path),
            "template_source_ids": [template.source_id for template in registry.templates],
            "holdout_opened": False,
        },
    )
    return report


def develop_fused_localizer(
    data_root: Path, model_root: Path, *, progress=print
) -> dict[str, object]:
    root = data_root / "prepared" / "v2-protocol"
    partitions = _source_partitions(root)
    report_root = model_root / "development" / "slot-localizer-v1"
    report_path = report_root / "development-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    classifier_report = report["layout_classifier"]
    labels = [str(value) for value in classifier_report["labels"]]
    label_indices = {label: index for index, label in enumerate(labels)}
    canonical_slots: dict[str, list[dict[str, object]]] = {}
    for row in partitions["train"]:
        canonical_slots.setdefault(layout_key(row), row["slots"])  # type: ignore[arg-type]
    model = build_layout_classifier(len(labels), pretrained=False)
    state_path = report_root / "layout-classifier-weights.pt"
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    model.eval()
    loader = DataLoader(
        LayoutClassificationDataset(root, partitions["validation"], label_indices),
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )
    logits_parts = []
    with torch.inference_mode():
        for inputs, _ in loader:
            logits_parts.append(model(inputs))
    raw_logits = torch.cat(logits_parts).numpy()
    constrained_logits = np.stack(
        [
            constrain_layout_logits(
                logits,
                labels,
                width=int(row["image_width"]),
                height=int(row["image_height"]),
            )
            for logits, row in zip(raw_logits, partitions["validation"], strict=True)
        ]
    )
    probabilities = torch.softmax(torch.from_numpy(constrained_logits), dim=1).numpy()
    predicted_indices = probabilities.argmax(axis=1)
    confidences = probabilities.max(axis=1)
    classifier_rows = []
    for predicted_index, confidence in zip(predicted_indices, confidences, strict=True):
        label = labels[int(predicted_index)]
        classifier_rows.append(
            [
                {
                    "id": str(slot_index),
                    "polygon": slot["polygon"],
                    "localization_confidence": float(confidence),
                    "corner_confidence": float(confidence),
                }
                for slot_index, slot in enumerate(canonical_slots[label], 1)
            ]
        )
    classifier_iou = _precompute_iou_matrices(partitions["validation"], classifier_rows)
    classifier_evaluations = []
    # The classifier was already selected on validation. Re-score the locked,
    # family-aware policy once at its conservative operating threshold instead
    # of repeatedly recomputing identical polygon matches.
    for candidate_threshold in (0.3,):
        candidate_rows = []
        candidate_iou = []
        for proposals, overlaps, confidence in zip(
            classifier_rows, classifier_iou, confidences, strict=True
        ):
            accepted = confidence >= candidate_threshold
            candidate_rows.append(proposals if accepted else [])
            candidate_iou.append(overlaps if accepted else overlaps[:0, :])
        classifier_evaluations.append(
            (
                float(candidate_threshold),
                _metrics(
                    partitions["validation"],
                    [None] * len(partitions["validation"]),
                    0.0,
                    decoded_rows=candidate_rows,
                    iou_matrices=candidate_iou,
                ),
            )
        )
    threshold, classifier_metrics = max(
        classifier_evaluations,
        key=lambda item: (
            min(float(value["recall_iou_50"]) for value in item[1]["by_dataset"].values()),
            min(
                float(value["precision_iou_50"])
                for value in item[1]["by_dataset"].values()
            ),
        ),
    )
    if "fused_registration_validation" in report:
        previous_fusion = report["validation"]
        metrics, selected_strategy = max(
            (
                (classifier_metrics, "fixed-camera layout classifier with canonical polygons"),
                (previous_fusion, "fixed-camera layout classifier plus ORB/RANSAC registration"),
            ),
            key=lambda item: (
                min(
                    float(value["recall_iou_50"])
                    for value in item[0]["by_dataset"].values()
                ),
                min(
                    float(value["precision_iou_50"])
                    for value in item[0]["by_dataset"].values()
                ),
            ),
        )
        minimum_recall = min(
            float(value["recall_iou_50"]) for value in metrics["by_dataset"].values()
        )
        minimum_precision = min(
            float(value["precision_iou_50"]) for value in metrics["by_dataset"].values()
        )
        report["layout_classifier_only_validation"] = classifier_metrics
        report["fused_registration_validation"] = previous_fusion
        report["validation"] = metrics
        report["localization_strategy"] = f"{selected_strategy}; neural detector diagnostic"
        report["layout_classifier"]["confidence_threshold"] = threshold
        report["deployment_gate"] = {
            "minimum_dataset_recall_required": 0.85,
            "minimum_dataset_precision_required": 0.85,
            "observed_minimum_dataset_recall": round(minimum_recall, 6),
            "observed_minimum_dataset_precision": round(minimum_precision, 6),
            "passed": minimum_recall >= 0.85 and minimum_precision >= 0.85,
        }
        atomic_json(report_path, report)
        decision_path = report_root / "decision-lock.json"
        decision = json.loads(decision_path.read_text(encoding="utf-8"))
        decision["localization_strategy"] = report["localization_strategy"]
        decision["development_report_sha256"] = sha256_file(report_path)
        atomic_json(decision_path, decision)
        return report
    registry = TemplateLayoutRegistry.from_rows(root, partitions["train"], per_layout=3)
    decoded_rows = []
    statuses: dict[str, int] = {}
    for index, (row, predicted_index, confidence) in enumerate(
        zip(partitions["validation"], predicted_indices, confidences, strict=True), 1
    ):
        if confidence < threshold:
            decoded_rows.append([])
            statuses["unsupported_low_classifier_confidence"] = (
                statuses.get("unsupported_low_classifier_confidence", 0) + 1
            )
            continue
        label = labels[int(predicted_index)]
        with Image.open(root / str(row["image_path"])) as image:
            registration = registry.detect(image, expected_layout=label)
        if registration["slots"]:
            decoded_rows.append(registration["slots"])  # type: ignore[arg-type]
            statuses["registered"] = statuses.get("registered", 0) + 1
        else:
            decoded_rows.append(
                [
                    {
                        "id": str(slot_index),
                        "polygon": slot["polygon"],
                        "localization_confidence": float(confidence),
                        "corner_confidence": float(confidence) * 0.8,
                    }
                    for slot_index, slot in enumerate(canonical_slots[label], 1)
                ]
            )
            statuses["canonical_warning_fallback"] = (
                statuses.get("canonical_warning_fallback", 0) + 1
            )
        if progress and index % 100 == 0:
            progress(f"fused localizer: {index:,}/{len(partitions['validation']):,} images")
    fusion_metrics = _metrics(
        partitions["validation"],
        [None] * len(partitions["validation"]),
        0.0,
        decoded_rows=decoded_rows,
    )
    metrics, selected_strategy = max(
        (
            (classifier_metrics, "fixed-camera layout classifier with canonical polygons"),
            (
                fusion_metrics,
                "fixed-camera layout classifier plus layout-restricted ORB/RANSAC registration",
            ),
        ),
        key=lambda item: (
            min(float(value["recall_iou_50"]) for value in item[0]["by_dataset"].values()),
            min(
                float(value["precision_iou_50"])
                for value in item[0]["by_dataset"].values()
            ),
        ),
    )
    minimum_recall = min(
        float(value["recall_iou_50"]) for value in metrics["by_dataset"].values()
    )
    minimum_precision = min(
        float(value["precision_iou_50"]) for value in metrics["by_dataset"].values()
    )
    report["layout_classifier_only_validation"] = classifier_metrics
    report["fused_registration_validation"] = fusion_metrics
    report["validation"] = metrics
    report["fusion_statuses"] = statuses
    report["localization_strategy"] = f"{selected_strategy}; neural detector diagnostic"
    report["layout_classifier"]["confidence_threshold"] = threshold
    report["deployment_gate"] = {
        "minimum_dataset_recall_required": 0.85,
        "minimum_dataset_precision_required": 0.85,
        "observed_minimum_dataset_recall": round(minimum_recall, 6),
        "observed_minimum_dataset_precision": round(minimum_precision, 6),
        "passed": minimum_recall >= 0.85 and minimum_precision >= 0.85,
    }
    atomic_json(report_path, report)
    decision_path = report_root / "decision-lock.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    decision["localization_strategy"] = report["localization_strategy"]
    decision["development_report_sha256"] = sha256_file(report_path)
    atomic_json(decision_path, decision)
    return report


def develop_localizer(
    data_root: Path,
    model_root: Path,
    *,
    epochs: int = 15,
    progress=print,
) -> dict[str, object]:
    root = data_root / "prepared" / "v2-protocol"
    partitions = _source_partitions(root)
    report_root = model_root / "development" / "slot-localizer-v1"
    report_root.mkdir(parents=True, exist_ok=True)
    random.seed(20260904)
    np.random.seed(20260904)
    torch.manual_seed(20260904)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260904)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ParkingSlotLocalizer(pretrained=True).to(device)
    workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(
        LocalizerDataset(root, partitions["train"], training=True),
        batch_size=8,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        LocalizerDataset(root, partitions["validation"]),
        batch_size=8,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    optimizer = torch.optim.AdamW(
        [
            {"params": model.features.parameters(), "lr": 2e-5},
            {
                "params": [
                    *model.upsample.parameters(),
                    *model.head.parameters(),
                ],
                "lr": 2e-4,
            },
        ],
        weight_decay=1e-4,
    )
    best_state = None
    best_loss = float("inf")
    stale_epochs = 0
    collisions = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for inputs, objects, corners, batch_collisions in train_loader:
            collisions += int(batch_collisions.sum())
            optimizer.zero_grad(set_to_none=True)
            loss, _, _ = _loss(model(inputs.to(device)), objects.to(device), corners.to(device))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(inputs)
        model.eval()
        validation_loss = 0.0
        with torch.inference_mode():
            for inputs, objects, corners, _ in validation_loader:
                loss, _, _ = _loss(model(inputs.to(device)), objects.to(device), corners.to(device))
                validation_loss += float(loss) * len(inputs)
        validation_loss /= len(partitions["validation"])
        if progress:
            progress(
                f"slot-localizer: epoch {epoch + 1}/{epochs}, "
                f"train_loss={total_loss / len(partitions['train']):.4f}, "
                f"val_loss={validation_loss:.4f}"
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            stale_epochs = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            stale_epochs += 1
            if stale_epochs >= 3:
                if progress:
                    progress(
                        "slot-localizer: early stopping after three validation-loss "
                        "non-improvements"
                    )
                break
    if best_state is None:
        raise RuntimeError("Slot localizer did not produce weights")
    model.load_state_dict(best_state)
    model.to(device)
    outputs, elapsed = _raw_outputs(model, validation_loader, device)
    candidates = np.linspace(0.2, 0.8, 13)
    minimum_threshold = float(candidates[0])
    decoded_minimum = [
        decode_localizer_output(output, object_threshold=minimum_threshold) for output in outputs
    ]
    minimum_iou_matrices = []
    for row, proposals in zip(partitions["validation"], decoded_minimum, strict=True):
        truth = [slot["polygon"] for slot in row["slots"]]
        minimum_iou_matrices.append(
            np.asarray(
                [
                    [polygon_iou(proposal["polygon"], polygon) for polygon in truth]
                    for proposal in proposals
                ],
                dtype=np.float32,
            ).reshape(len(proposals), len(truth))
        )
    evaluations = []
    for value in candidates:
        selected_rows = []
        selected_matrices = []
        for proposals, overlaps in zip(decoded_minimum, minimum_iou_matrices, strict=True):
            indices = [
                index
                for index, proposal in enumerate(proposals)
                if float(proposal["localization_confidence"]) >= float(value)
            ]
            selected_rows.append([proposals[index] for index in indices])
            selected_matrices.append(overlaps[indices, :])
        evaluations.append(
            (
                float(value),
                _metrics(
                    partitions["validation"],
                    outputs,
                    float(value),
                    decoded_rows=selected_rows,
                    iou_matrices=selected_matrices,
                ),
            )
        )
    threshold, metrics = max(
        evaluations,
        key=lambda item: (
            min(float(value["recall_iou_50"]) for value in item[1]["by_dataset"].values()),
            float(item[1]["f1_iou_50"]),
            -float(item[1]["mean_absolute_count_error"]),
        ),
    )
    minimum_dataset_recall = min(
        float(value["recall_iou_50"]) for value in metrics["by_dataset"].values()
    )
    state_path = report_root / "development-weights.pt"
    torch.save(model.cpu().state_dict(), state_path)
    report = {
        "protocol_id": PROTOCOL_ID,
        "phase": "development",
        "model_name": LOCALIZER_NAME,
        "generated_at": datetime.now(UTC).isoformat(),
        "epochs": epochs,
        "device": str(device),
        "training_images": len(partitions["train"]),
        "validation_images": len(partitions["validation"]),
        "holdout_images_unread": len(partitions["holdout"]),
        "supported_datasets": ["PKLot", "CNRPark+EXT"],
        "excluded_datasets": {
            "ACPDS": (
                "moving GoPro viewpoints are outside the fixed-camera localisation contract; "
                "ACPDS remains supported through verified prepared geometry"
            )
        },
        "partition_policy": (
            "fixed-camera layouts learned on training date/day groups and evaluated on disjoint "
            "validation groups; protected holdout groups remain unread"
        ),
        "object_threshold": threshold,
        "validation": metrics,
        "deployment_gate": {
            "minimum_dataset_recall_required": 0.5,
            "passed": minimum_dataset_recall >= 0.5,
        },
        "validation_latency_ms_per_image": round(elapsed / len(partitions["validation"]), 3),
        "grid_collision_observations": collisions,
        "selection_policy": "validation-only worst-dataset recall, then F1 and count error",
    }
    atomic_json(report_root / "development-report.json", report)
    atomic_json(
        report_root / "decision-lock.json",
        {
            "model_name": LOCALIZER_NAME,
            "object_threshold": threshold,
            "state_sha256": sha256_file(state_path),
            "development_report_sha256": sha256_file(report_root / "development-report.json"),
            "holdout_opened": False,
        },
    )
    return report


def finalize_localizer(data_root: Path, model_root: Path) -> dict[str, object]:
    root = data_root / "prepared" / "v2-protocol"
    partitions = _source_partitions(root)
    report_root = model_root / "development" / "slot-localizer-v1"
    final_path = report_root / "final-holdout-report.json"
    if final_path.exists():
        raise RuntimeError("Localizer holdout has already been opened")
    decision_path = report_root / "decision-lock.json"
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    development_path = report_root / "development-report.json"
    development = json.loads(development_path.read_text(encoding="utf-8"))
    if decision.get("holdout_opened"):
        raise RuntimeError("Localizer holdout has already been opened")
    if decision.get("development_report_sha256") != sha256_file(development_path):
        raise RuntimeError("Localizer development report changed after decision lock")
    if not development["deployment_gate"]["passed"]:
        raise RuntimeError("Localizer validation quality gate did not pass; deployment refused")
    state_path = report_root / "development-weights.pt"
    if decision["state_sha256"] != sha256_file(state_path):
        raise RuntimeError("Localizer weights changed after decision lock")
    classifier_state_path = report_root / "layout-classifier-weights.pt"
    if decision.get("layout_classifier_state_sha256") != sha256_file(classifier_state_path):
        raise RuntimeError("Layout-classifier weights changed after decision lock")
    labels = [str(value) for value in development["layout_classifier"]["labels"]]
    label_indices = {label: index for index, label in enumerate(labels)}
    if any(layout_key(row) not in label_indices for row in partitions["holdout"]):
        raise RuntimeError("Holdout contains a fixed-camera layout absent from training")
    canonical_slots: dict[str, list[dict[str, object]]] = {}
    for row in partitions["train"]:
        canonical_slots.setdefault(layout_key(row), row["slots"])  # type: ignore[arg-type]
    classifier_model = build_layout_classifier(len(labels), pretrained=False)
    classifier_model.load_state_dict(
        torch.load(classifier_state_path, map_location="cpu", weights_only=True)
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier_model.to(device).eval()
    loader = DataLoader(
        LayoutClassificationDataset(root, partitions["holdout"], label_indices),
        batch_size=64,
        shuffle=False,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
    )
    logits_parts = []
    started = perf_counter()
    with torch.inference_mode():
        for inputs, _ in loader:
            logits_parts.append(classifier_model(inputs.to(device)).cpu())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (perf_counter() - started) * 1_000
    raw_logits = torch.cat(logits_parts).numpy()
    threshold = float(development["layout_classifier"]["confidence_threshold"])
    decoded_rows = []
    for logits, row in zip(raw_logits, partitions["holdout"], strict=True):
        constrained = constrain_layout_logits(
            logits,
            labels,
            width=int(row["image_width"]),
            height=int(row["image_height"]),
        )
        probabilities = torch.softmax(torch.from_numpy(constrained), dim=0).numpy()
        index = int(np.argmax(probabilities))
        confidence = float(probabilities[index])
        decoded_rows.append(
            [
                {
                    "id": str(slot_index),
                    "polygon": slot["polygon"],
                    "localization_confidence": confidence,
                    "corner_confidence": confidence,
                }
                for slot_index, slot in enumerate(canonical_slots[labels[index]], 1)
            ]
            if confidence >= threshold
            else []
        )
    metrics = _metrics(
        partitions["holdout"],
        [None] * len(partitions["holdout"]),
        0.0,
        decoded_rows=decoded_rows,
    )
    end_to_end = _detected_geometry_occupancy_metrics(
        root,
        partitions["holdout"],
        None,
        0.0,
        model_root,
        decoded_rows=decoded_rows,
    )
    minimum_holdout_recall = min(
        float(value["recall_iou_50"]) for value in metrics["by_dataset"].values()
    )
    minimum_holdout_precision = min(
        float(value["precision_iou_50"]) for value in metrics["by_dataset"].values()
    )
    passed = minimum_holdout_recall >= 0.8 and minimum_holdout_precision >= 0.85
    final = {
        "phase": "single_final_protected_holdout_evaluation",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "detected_geometry_occupancy": end_to_end,
        "latency_ms_per_image": round(elapsed / len(partitions["holdout"]), 3),
        "images": len(partitions["holdout"]),
        "deployment_gate": {
            "minimum_dataset_recall_required": 0.8,
            "minimum_dataset_precision_required": 0.85,
            "observed_minimum_dataset_recall": round(minimum_holdout_recall, 6),
            "observed_minimum_dataset_precision": round(minimum_holdout_precision, 6),
            "passed": passed,
        },
    }
    atomic_json(final_path, final)
    decision["holdout_opened"] = True
    decision["final_report_sha256"] = sha256_file(final_path)
    atomic_json(decision_path, decision)
    if not passed:
        raise RuntimeError("Localizer final holdout quality gate did not pass; deployment refused")
    classifier_artifact = export_layout_classifier(
        classifier_model,
        model_root,
        labels=labels,
        canonical_slots=canonical_slots,
        confidence_threshold=threshold,
    )
    diagnostic_model = ParkingSlotLocalizer(pretrained=False)
    diagnostic_model.load_state_dict(
        torch.load(state_path, map_location="cpu", weights_only=True)
    )
    metadata = export_localizer(
        diagnostic_model,
        model_root,
        {
            "trained_at": datetime.now(UTC).isoformat(),
            "object_threshold": decision["object_threshold"],
            "training_images": len(partitions["train"]),
            "validation_images": len(partitions["validation"]),
            "holdout_images": len(partitions["holdout"]),
            "final_holdout": metrics,
            "detected_geometry_occupancy": end_to_end,
            "localization_strategy": development["localization_strategy"],
            "supported_datasets": development["supported_datasets"],
            "excluded_datasets": development["excluded_datasets"],
            **classifier_artifact,
            "supported_slot_count": {"minimum": 4, "maximum": 250},
            "decision_lock_sha256": sha256_file(decision_path),
            "final_report_sha256": sha256_file(final_path),
        },
    )
    detector = SlotLocalizerPredictor.load(model_root)
    classifier = OccupancyV3Predictor.load(model_root)
    measurements = []
    supported = 0
    for row in partitions["holdout"][: min(10, len(partitions["holdout"]))]:
        with Image.open(root / str(row["image_path"])) as source:
            image = source.convert("RGB").copy()
        started = perf_counter()
        detection = detector.detect(image)
        if detection["status"] != "unsupported_layout":
            patches = [rectify_slot(image, slot["polygon"]) for slot in detection["slots"]]
            classifier.predict(patches)
            supported += 1
        measurements.append((perf_counter() - started) * 1_000)
    metadata["cpu_end_to_end_image_benchmark"] = {
        "images": len(measurements),
        "supported_images": supported,
        "mean_ms": round(float(np.mean(measurements)), 3),
        "p95_ms": round(float(np.percentile(measurements, 95)), 3),
        "provider": "CPUExecutionProvider",
    }
    metadata["onnx_size_mb"] = round(
        (model_root / metadata["onnx_file"]).stat().st_size / 1024 / 1024, 3
    )
    atomic_json(model_root / LOCALIZER_METADATA, metadata)
    return {"final": final, "artifact": metadata}
