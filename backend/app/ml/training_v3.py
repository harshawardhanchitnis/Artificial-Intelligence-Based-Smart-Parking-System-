from __future__ import annotations

import json
import os
import platform
import random
import warnings
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil
import sklearn
import torch
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import balanced_accuracy_score, brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.datasets.integrity import PROTOCOL_ID, atomic_json, read_jsonl, sha256_file
from app.datasets.v2_protocol import prepare_v2_protocol
from app.ml.features import extract_features
from app.ml.model_store import ModelNotReadyError, load_model
from app.ml.occupancy_v3 import (
    OCCUPANCY_METADATA,
    OccupancyV3Predictor,
    build_occupancy_model,
    export_occupancy_model,
    preprocessing,
)
from app.ml.training import classification_metrics

DEVELOPMENT_REPORT = "development-report.json"
DECISION_LOCK = "decision-lock.json"
FINAL_REPORT = "final-holdout-report.json"
STATE_FILE = "selected-development-weights.pt"


class OccupancyDataset(Dataset):
    def __init__(self, root: Path, rows: list[dict[str, object]], *, training: bool) -> None:
        self.root = root
        self.rows = rows
        self.transform = preprocessing(training)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        with Image.open(self.root / str(row["patch_path"])) as image:
            inputs = self.transform(image.convert("RGB"))
        return inputs, torch.tensor(float(row["label"]), dtype=torch.float32)


def _seed(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _rows(protocol_root: Path) -> dict[str, list[dict[str, object]]]:
    manifest = protocol_root / "occupancy-manifest.jsonl"
    if not manifest.is_file():
        raise RuntimeError("Prepare the V2 leakage-safe protocol before model development")
    rows = read_jsonl(manifest)
    result = {
        name: [row for row in rows if row["partition"] == name]
        for name in ("train", "validation", "holdout")
    }
    if any(not partition for partition in result.values()):
        raise RuntimeError("All three protocol partitions are required")
    return result


def _probability_metrics(
    rows: list[dict[str, object]], probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    labels = np.asarray([int(row["label"]) for row in rows])
    predictions = (probabilities >= threshold).astype(np.int64)
    metrics = classification_metrics(labels, predictions)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    confidences = np.maximum(probabilities, 1 - probabilities)
    correct = predictions == labels
    for lower, upper in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidences >= lower) & (confidences < upper if upper < 1 else confidences <= upper)
        if np.any(mask):
            ece += float(np.mean(mask)) * abs(
                float(np.mean(confidences[mask])) - float(np.mean(correct[mask]))
            )
    by_dataset = {}
    by_group = {}
    for key_name, destination in (("dataset", by_dataset), ("group_id", by_group)):
        for value in sorted({str(row[key_name]) for row in rows}):
            indices = np.asarray(
                [index for index, row in enumerate(rows) if str(row[key_name]) == value]
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                destination[value] = classification_metrics(labels[indices], predictions[indices])
    matrix = metrics["confusion_matrix"]
    occupied = int(matrix["true_occupied"]) + int(matrix["false_vacant"])
    vacant = int(matrix["true_vacant"]) + int(matrix["false_occupied"])
    return {
        **metrics,
        "false_vacant_rate": round(int(matrix["false_vacant"]) / max(occupied, 1), 6),
        "false_occupied_rate": round(int(matrix["false_occupied"]) / max(vacant, 1), 6),
        "brier_score": round(float(brier_score_loss(labels, probabilities)), 6),
        "expected_calibration_error": round(ece, 6),
        "samples": len(rows),
        "by_dataset": by_dataset,
        "worst_group_balanced_accuracy": round(
            min(float(value["balanced_accuracy"]) for value in by_group.values()), 6
        ),
        "worst_groups": sorted(
            ({"group": key, **value} for key, value in by_group.items()),
            key=lambda value: float(value["balanced_accuracy"]),
        )[:10],
    }


def _select_threshold(rows: list[dict[str, object]], probabilities: np.ndarray) -> float:
    candidates = np.linspace(0.2, 0.8, 121)
    labels = np.asarray([int(row["label"]) for row in rows])
    datasets = sorted({str(row["dataset"]) for row in rows})

    def score(threshold: float) -> tuple[float, float, float]:
        predictions = probabilities >= threshold
        minimum = min(
            float(
                balanced_accuracy_score(
                    labels[[index for index, row in enumerate(rows) if row["dataset"] == dataset]],
                    predictions[
                        [index for index, row in enumerate(rows) if row["dataset"] == dataset]
                    ],
                )
            )
            for dataset in datasets
        )
        overall = float(balanced_accuracy_score(labels, predictions))
        false_vacant = float(np.mean(~predictions[labels == 1]))
        return (
            minimum,
            overall - 0.25 * false_vacant,
            -abs(float(threshold) - 0.5),
        )

    return float(max(candidates, key=score))


def _temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    logits_tensor = torch.tensor(logits, dtype=torch.float64)
    labels_tensor = torch.tensor(labels, dtype=torch.float64)
    log_temperature = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=75)
    criterion = nn.BCEWithLogitsLoss()

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = criterion(logits_tensor / torch.exp(log_temperature), labels_tensor)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(torch.exp(log_temperature).detach().clamp(0.25, 4.0))


def _infer(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, float]:
    outputs = []
    model.eval()
    started = perf_counter()
    with torch.inference_mode():
        for inputs, _ in loader:
            outputs.append(model(inputs.to(device)).detach().cpu().numpy())
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = (perf_counter() - started) * 1_000
    return np.concatenate(outputs), elapsed


def _train_neural(
    architecture: str,
    protocol_root: Path,
    partitions: dict[str, list[dict[str, object]]],
    device: torch.device,
    *,
    epochs: int,
    batch_size: int,
    pretrained: bool,
    progress,
) -> tuple[nn.Module, np.ndarray, dict[str, object]]:
    model = build_occupancy_model(architecture, pretrained=pretrained).to(device)
    workers = 2 if device.type == "cuda" else 0
    train_loader = DataLoader(
        OccupancyDataset(protocol_root, partitions["train"], training=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    validation_loader = DataLoader(
        OccupancyDataset(protocol_root, partitions["validation"], training=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_state = None
    best_loss = float("inf")
    started = perf_counter()
    for epoch in range(epochs):
        model.train()
        running = 0.0
        for inputs, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(labels)
        validation_logits, _ = _infer(model, validation_loader, device)
        validation_labels = np.asarray([int(row["label"]) for row in partitions["validation"]])
        validation_loss = float(
            nn.functional.binary_cross_entropy_with_logits(
                torch.tensor(validation_logits),
                torch.tensor(validation_labels, dtype=torch.float32),
            )
        )
        if progress:
            progress(
                f"{architecture}: epoch {epoch + 1}/{epochs}, "
                f"train_loss={running / len(partitions['train']):.4f}, "
                f"val_loss={validation_loss:.4f}"
            )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError(f"{architecture} did not produce weights")
    model.load_state_dict(best_state)
    model.to(device)
    logits, elapsed = _infer(model, validation_loader, device)
    return (
        model,
        logits,
        {
            "training_seconds": round(perf_counter() - started, 3),
            "validation_latency_ms_per_slot": round(elapsed / len(partitions["validation"]), 4),
            "parameters": sum(parameter.numel() for parameter in model.parameters()),
            "estimated_fp32_mb": round(
                sum(parameter.numel() for parameter in model.parameters()) * 4 / 1024 / 1024,
                3,
            ),
        },
    )


def _classical_candidate(
    protocol_root: Path,
    partitions: dict[str, list[dict[str, object]]],
    *,
    cap: int,
    progress,
) -> dict[str, object]:
    train_rows = partitions["train"][:cap]
    validation_rows = partitions["validation"]
    features = []
    labels = []
    ordered = train_rows + validation_rows
    started = perf_counter()
    for index, row in enumerate(ordered, start=1):
        with Image.open(protocol_root / str(row["patch_path"])) as image:
            features.append(extract_features(image.convert("RGB")))
        labels.append(int(row["label"]))
        if progress and index % 2_000 == 0:
            progress(f"linear-svm features: {index:,}/{len(ordered):,}")
    array = np.asarray(features)
    labels_array = np.asarray(labels)
    scaler = StandardScaler().fit(array[: len(train_rows)])
    classifier = LinearSVC(C=1.0, class_weight="balanced", random_state=20260904)
    classifier.fit(scaler.transform(array[: len(train_rows)]), labels_array[: len(train_rows)])
    calibrator = CalibratedClassifierCV(FrozenEstimator(classifier), method="sigmoid")
    calibrator.fit(scaler.transform(array[len(train_rows) :]), labels_array[len(train_rows) :])
    probabilities = calibrator.predict_proba(scaler.transform(array[len(train_rows) :]))[:, 1]
    threshold = _select_threshold(validation_rows, probabilities)
    learning_curve = []
    for sample_count in sorted({min(len(train_rows), value) for value in (2_000, 8_000, cap)}):
        curve_scaler = StandardScaler().fit(array[:sample_count])
        curve_model = LinearSVC(C=1.0, class_weight="balanced", random_state=20260904)
        curve_model.fit(curve_scaler.transform(array[:sample_count]), labels_array[:sample_count])
        curve_calibrator = CalibratedClassifierCV(FrozenEstimator(curve_model), method="sigmoid")
        curve_calibrator.fit(
            curve_scaler.transform(array[len(train_rows) :]), labels_array[len(train_rows) :]
        )
        curve_probabilities = curve_calibrator.predict_proba(
            curve_scaler.transform(array[len(train_rows) :])
        )[:, 1]
        curve_threshold = _select_threshold(validation_rows, curve_probabilities)
        curve_metrics = _probability_metrics(validation_rows, curve_probabilities, curve_threshold)
        learning_curve.append(
            {
                "training_samples": sample_count,
                "balanced_accuracy": curve_metrics["balanced_accuracy"],
                "worst_group_balanced_accuracy": curve_metrics["worst_group_balanced_accuracy"],
                "false_vacant_rate": curve_metrics["false_vacant_rate"],
            }
        )
    return {
        "architecture": "hog-color-linear-svm",
        "threshold": threshold,
        "temperature": None,
        "validation": _probability_metrics(validation_rows, probabilities, threshold),
        "training_seconds": round(perf_counter() - started, 3),
        "validation_latency_ms_per_slot": None,
        "parameters": int(classifier.coef_.size),
        "estimated_fp32_mb": round(classifier.coef_.size * 4 / 1024 / 1024, 3),
        "learning_curve": learning_curve,
        "deployment_note": "comparison candidate; neural winner is exported for deployment",
    }


def _frozen_baseline_candidate(
    protocol_root: Path,
    validation_rows: list[dict[str, object]],
    model_root: Path,
    *,
    progress,
) -> dict[str, object]:
    try:
        baseline = load_model(model_root)
    except ModelNotReadyError as exc:
        return {
            "architecture": "parking-occupancy-logistic-v2",
            "available": False,
            "reason": str(exc),
            "deployment_note": "historical artifact remains preserved; no V2 holdout access",
        }
    started = perf_counter()
    features = []
    for index, row in enumerate(validation_rows, 1):
        with Image.open(protocol_root / str(row["patch_path"])) as image:
            features.append(extract_features(image.convert("RGB")))
        if progress and index % 2_000 == 0:
            progress(f"frozen logistic-v2 validation: {index:,}/{len(validation_rows):,}")
    array = np.asarray(features, dtype=np.float64)
    normalized = (array - baseline.feature_mean) / baseline.feature_scale
    logits = np.clip(normalized @ baseline.coefficient + baseline.intercept, -40, 40)
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return {
        "architecture": "parking-occupancy-logistic-v2",
        "available": True,
        "threshold": baseline.threshold,
        "validation": _probability_metrics(validation_rows, probabilities, baseline.threshold),
        "validation_seconds": round(perf_counter() - started, 3),
        "frozen": True,
        "deployment_note": (
            "reproducible historical baseline evaluated without fitting, calibration, or retuning"
        ),
    }


def develop_occupancy_v3(
    data_root: Path,
    model_root: Path,
    *,
    profile: str = "standard",
    epochs: int | None = None,
    progress=print,
) -> dict[str, object]:
    protocol_root = data_root / "prepared" / "v2-protocol"
    if not (protocol_root / "protocol-report.json").is_file():
        prepare_v2_protocol(data_root, profile=profile, progress=progress)
    partitions = _rows(protocol_root)
    report_root = model_root / "development" / "occupancy-v3"
    report_root.mkdir(parents=True, exist_ok=True)
    _seed(20260904)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_count = epochs or (2 if profile == "smoke" else 8)
    batch_size = 32 if device.type == "cpu" else 96
    candidates = [
        _frozen_baseline_candidate(
            protocol_root,
            partitions["validation"],
            model_root,
            progress=progress,
        ),
        _classical_candidate(
            protocol_root,
            partitions,
            cap=2_000 if profile == "smoke" else 24_000,
            progress=progress,
        ),
    ]
    trained_models: dict[str, nn.Module] = {}
    for architecture in ("compact-cnn", "mobilenet-v3-small", "efficientnet-b0"):
        model, logits, runtime = _train_neural(
            architecture,
            protocol_root,
            partitions,
            device,
            epochs=epoch_count,
            batch_size=batch_size,
            pretrained=architecture != "compact-cnn",
            progress=progress,
        )
        temperature = _temperature(
            logits, np.asarray([int(row["label"]) for row in partitions["validation"]])
        )
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -40, 40)))
        threshold = _select_threshold(partitions["validation"], probabilities)
        candidate = {
            "architecture": architecture,
            "threshold": threshold,
            "temperature": temperature,
            "validation": _probability_metrics(partitions["validation"], probabilities, threshold),
            **runtime,
        }
        candidates.append(candidate)
        trained_models[architecture] = model.cpu()

    neural = [candidate for candidate in candidates if candidate["architecture"] in trained_models]
    selected = max(
        neural,
        key=lambda candidate: (
            float(candidate["validation"]["worst_group_balanced_accuracy"]),
            min(
                float(metrics["balanced_accuracy"])
                for metrics in candidate["validation"]["by_dataset"].values()
            ),
            float(candidate["validation"]["balanced_accuracy"]),
            -float(candidate["validation_latency_ms_per_slot"]),
        ),
    )
    state_path = report_root / STATE_FILE
    torch.save(trained_models[str(selected["architecture"])].state_dict(), state_path)
    protocol_report = json.loads(
        (protocol_root / "protocol-report.json").read_text(encoding="utf-8")
    )
    selected_validation = selected["validation"]
    minimum_dataset_balanced = min(
        float(value["balanced_accuracy"]) for value in selected_validation["by_dataset"].values()
    )
    validation_gate_passed = (
        float(selected_validation["balanced_accuracy"]) >= 0.9
        and minimum_dataset_balanced >= 0.85
        and float(selected_validation["false_vacant_rate"]) <= 0.15
    )
    development = {
        "protocol_id": PROTOCOL_ID,
        "phase": "development",
        "generated_at": datetime.now(UTC).isoformat(),
        "profile": profile,
        "device": str(device),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "selection_objective": (
            "worst-group, then worst-dataset, then overall balanced accuracy, then latency"
        ),
        "selected_architecture": selected["architecture"],
        "selected_threshold": selected["threshold"],
        "selected_temperature": selected["temperature"],
        "deployment_gate": {
            "overall_balanced_accuracy_required": 0.9,
            "minimum_dataset_balanced_accuracy_required": 0.85,
            "maximum_false_vacant_rate": 0.15,
            "observed_minimum_dataset_balanced_accuracy": round(minimum_dataset_balanced, 6),
            "passed": validation_gate_passed,
        },
        "training_samples": len(partitions["train"]),
        "validation_samples": len(partitions["validation"]),
        "holdout_samples_unread": len(partitions["holdout"]),
        "protocol_report_sha256": sha256_file(protocol_root / "protocol-report.json"),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "sklearn": sklearn.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
    }
    atomic_json(report_root / DEVELOPMENT_REPORT, development)
    decision = {
        "protocol_id": PROTOCOL_ID,
        "architecture": selected["architecture"],
        "threshold": selected["threshold"],
        "temperature": selected["temperature"],
        "preprocessing": protocol_report["preprocessing"],
        "development_report_sha256": sha256_file(report_root / DEVELOPMENT_REPORT),
        "state_sha256": sha256_file(state_path),
        "locked_at": datetime.now(UTC).isoformat(),
        "holdout_opened": False,
    }
    atomic_json(report_root / DECISION_LOCK, decision)
    return development


def finalize_occupancy_v3(data_root: Path, model_root: Path) -> dict[str, object]:
    protocol_root = data_root / "prepared" / "v2-protocol"
    report_root = model_root / "development" / "occupancy-v3"
    final_path = report_root / FINAL_REPORT
    if final_path.exists():
        raise RuntimeError(
            "Final protected holdout has already been opened; refusing repeated evaluation"
        )
    decision_path = report_root / DECISION_LOCK
    state_path = report_root / STATE_FILE
    if not decision_path.is_file() or not state_path.is_file():
        raise RuntimeError("Complete and lock model development before final holdout evaluation")
    decision = json.loads(decision_path.read_text(encoding="utf-8"))
    if decision.get("holdout_opened"):
        raise RuntimeError("Decision lock shows that the protected holdout was already opened")
    if decision["state_sha256"] != sha256_file(state_path):
        raise RuntimeError("Selected weights changed after the development decision lock")
    if decision["development_report_sha256"] != sha256_file(report_root / DEVELOPMENT_REPORT):
        raise RuntimeError("Development report changed after the decision lock")
    development = json.loads((report_root / DEVELOPMENT_REPORT).read_text(encoding="utf-8"))
    if not development["deployment_gate"]["passed"]:
        raise RuntimeError("Occupancy validation quality gate did not pass; deployment refused")
    partitions = _rows(protocol_root)
    model = build_occupancy_model(str(decision["architecture"]), pretrained=False)
    model.load_state_dict(torch.load(state_path, map_location="cpu", weights_only=True))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    loader = DataLoader(
        OccupancyDataset(protocol_root, partitions["holdout"], training=False),
        batch_size=96 if device.type == "cuda" else 32,
        shuffle=False,
        num_workers=2 if device.type == "cuda" else 0,
        pin_memory=device.type == "cuda",
    )
    logits, elapsed = _infer(model, loader, device)
    temperature = float(decision["temperature"])
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logits / temperature, -40, 40)))
    metrics = _probability_metrics(
        partitions["holdout"], probabilities, float(decision["threshold"])
    )
    minimum_dataset_balanced = min(
        float(value["balanced_accuracy"]) for value in metrics["by_dataset"].values()
    )
    final_gate_passed = (
        float(metrics["balanced_accuracy"]) >= 0.88
        and minimum_dataset_balanced >= 0.8
        and float(metrics["false_vacant_rate"]) <= 0.2
    )
    final = {
        "protocol_id": PROTOCOL_ID,
        "phase": "single_final_protected_holdout_evaluation",
        "evaluated_at": datetime.now(UTC).isoformat(),
        "architecture": decision["architecture"],
        "threshold": decision["threshold"],
        "temperature": temperature,
        "metrics": metrics,
        "latency_ms_per_slot": round(elapsed / len(partitions["holdout"]), 4),
        "decision_lock_sha256": sha256_file(decision_path),
        "holdout_samples": len(partitions["holdout"]),
        "deployment_gate": {
            "overall_balanced_accuracy_required": 0.88,
            "minimum_dataset_balanced_accuracy_required": 0.8,
            "maximum_false_vacant_rate": 0.2,
            "observed_minimum_dataset_balanced_accuracy": round(minimum_dataset_balanced, 6),
            "passed": final_gate_passed,
        },
    }
    atomic_json(final_path, final)
    decision["holdout_opened"] = True
    decision["holdout_report_sha256"] = sha256_file(final_path)
    atomic_json(decision_path, decision)
    if not final_gate_passed:
        raise RuntimeError("Occupancy final holdout quality gate did not pass; deployment refused")
    process = psutil.Process(os.getpid())
    metadata = export_occupancy_model(
        model,
        model_root,
        architecture=str(decision["architecture"]),
        threshold=float(decision["threshold"]),
        temperature=temperature,
        metadata={
            "trained_at": decision["locked_at"],
            "training_samples": len(partitions["train"]),
            "validation_samples": len(partitions["validation"]),
            "final_holdout_samples": len(partitions["holdout"]),
            "selection_policy": "validation only; worst-group priority",
            "calibration": {
                "method": "temperature scaling",
                "validation_only": True,
                "temperature": temperature,
            },
            "final_holdout": metrics,
            "observed_process_rss_mb": round(process.memory_info().rss / 1024 / 1024, 2),
            "development_report_sha256": sha256_file(report_root / DEVELOPMENT_REPORT),
            "final_report_sha256": sha256_file(final_path),
        },
    )
    cpu_predictor = OccupancyV3Predictor.load(model_root)
    benchmark_rows = partitions["holdout"][: min(512, len(partitions["holdout"]))]
    benchmark_images = []
    for row in benchmark_rows:
        with Image.open(protocol_root / str(row["patch_path"])) as image:
            benchmark_images.append(image.convert("RGB").copy())
    rss_before = process.memory_info().rss
    cpu_started = perf_counter()
    _, cpu_onnx_ms = cpu_predictor.probabilities(benchmark_images)
    cpu_wall_ms = (perf_counter() - cpu_started) * 1_000
    cpu_benchmark = {
        "samples": len(benchmark_images),
        "onnx_execution_ms_per_slot": round(cpu_onnx_ms / len(benchmark_images), 4),
        "end_to_end_batch_ms_per_slot": round(cpu_wall_ms / len(benchmark_images), 4),
        "rss_delta_mb": round((process.memory_info().rss - rss_before) / 1024 / 1024, 2),
        "provider": "CPUExecutionProvider",
    }
    metadata["cpu_benchmark"] = cpu_benchmark
    metadata["onnx_size_mb"] = round(
        (model_root / metadata["onnx_file"]).stat().st_size / 1024 / 1024, 3
    )
    metadata["int8_decision"] = (
        "FP32 retained: convolution-heavy transfer models receive limited dynamic-INT8 benefit; "
        "quantized deployment requires a separate validation-locked static calibration study"
    )
    atomic_json(model_root / OCCUPANCY_METADATA, metadata)
    return {"final": final, "artifact": metadata}
