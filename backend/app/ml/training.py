from __future__ import annotations

import json
import platform
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import sklearn
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler

from app.datasets.benchmark import PARTITIONS, verify_benchmark
from app.ml.features import FEATURE_VERSION, extract_features
from app.ml.model_store import MODEL_NAME, save_model


class TrainingError(RuntimeError):
    """Raised when a valid occupancy model cannot be trained."""


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    true_vacant, false_occupied = (int(value) for value in matrix[0])
    false_vacant, true_occupied = (int(value) for value in matrix[1])
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, predictions)), 6),
        "precision_occupied": round(
            float(precision_score(labels, predictions, zero_division=0)), 6
        ),
        "recall_occupied": round(float(recall_score(labels, predictions, zero_division=0)), 6),
        "f1_occupied": round(float(f1_score(labels, predictions, zero_division=0)), 6),
        "specificity_vacant": round(
            true_vacant / (true_vacant + false_occupied) if true_vacant + false_occupied else 0.0,
            6,
        ),
        "confusion_matrix": {
            "true_vacant": true_vacant,
            "false_occupied": false_occupied,
            "false_vacant": false_vacant,
            "true_occupied": true_occupied,
        },
    }


def _fit(features: np.ndarray, labels: np.ndarray, random_state: int) -> tuple:
    scaler = StandardScaler()
    normalized = scaler.fit_transform(features)
    classifier = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=3_000,
        random_state=random_state,
        solver="liblinear",
    )
    classifier.fit(normalized, labels)
    if classifier.classes_.tolist() != [0, 1]:
        raise TrainingError("Training data must contain vacant and occupied classes")
    return scaler, classifier


def _load_rows(data_root: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    benchmark_root = data_root / "prepared" / "ml-benchmark"
    verification = verify_benchmark(data_root)
    report = json.loads((benchmark_root / "report.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(line)
        for line in (benchmark_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    return rows, {**report, "verification": verification}


def _extract(
    data_root: Path, rows: list[dict[str, object]], progress
) -> tuple[np.ndarray, np.ndarray]:
    root = data_root / "prepared" / "ml-benchmark"
    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    for index, row in enumerate(rows, start=1):
        with Image.open(root / str(row["patch_path"])) as image:
            feature_rows.append(extract_features(image.convert("RGB")))
        labels.append(int(row["label"]))
        if progress and index % 500 == 0:
            progress(f"Features: {index:,}/{len(rows):,} benchmark samples")
    return np.asarray(feature_rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def _predict(
    scaler: StandardScaler,
    classifier: LogisticRegression,
    features: np.ndarray,
    threshold: float,
) -> np.ndarray:
    probabilities = classifier.predict_proba(scaler.transform(features))[:, 1]
    return (probabilities >= threshold).astype(np.int64)


def _evaluation(
    rows: list[dict[str, object]], labels: np.ndarray, predictions: np.ndarray
) -> dict[str, object]:
    metrics = classification_metrics(labels, predictions)
    majority = max(int(np.sum(labels == 0)), int(np.sum(labels == 1))) / len(labels)
    return {
        **metrics,
        "unique_samples": len(rows),
        "unique_sources": len({(row["dataset"], row["source_id"]) for row in rows}),
        "unique_groups": len({(row["dataset"], row["group_id"]) for row in rows}),
        "class_counts": {
            "vacant": int(np.sum(labels == 0)),
            "occupied": int(np.sum(labels == 1)),
        },
        "majority_baseline_accuracy": round(float(majority), 6),
    }


def train_model(
    data_root: Path,
    model_root: Path,
    *,
    random_state: int = 42,
    progress=print,
) -> dict[str, object]:
    rows, benchmark_report = _load_rows(data_root)
    by_partition = {
        partition: [row for row in rows if row["partition"] == partition]
        for partition in PARTITIONS
    }
    minimums = {"train": 10, "validation": 4, "test": 4}
    if any(len(by_partition[name]) < minimums[name] for name in PARTITIONS):
        raise TrainingError("Benchmark partitions do not contain enough samples")

    ordered = [row for partition in PARTITIONS for row in by_partition[partition]]
    features, labels = _extract(data_root, ordered, progress)
    offsets: dict[str, slice] = {}
    start = 0
    for partition in PARTITIONS:
        stop = start + len(by_partition[partition])
        offsets[partition] = slice(start, stop)
        start = stop

    train_slice = offsets["train"]
    scaler, classifier = _fit(features[train_slice], labels[train_slice], random_state)
    validation_slice = offsets["validation"]
    validation_probabilities = classifier.predict_proba(
        scaler.transform(features[validation_slice])
    )[:, 1]
    threshold_candidates = np.linspace(0.2, 0.8, 25)
    threshold = max(
        threshold_candidates,
        key=lambda value: balanced_accuracy_score(
            labels[validation_slice], validation_probabilities >= value
        ),
    )

    validation_predictions = (validation_probabilities >= threshold).astype(np.int64)
    validation = _evaluation(
        by_partition["validation"], labels[validation_slice], validation_predictions
    )
    test_slice = offsets["test"]
    test_predictions = _predict(scaler, classifier, features[test_slice], float(threshold))
    test = _evaluation(by_partition["test"], labels[test_slice], test_predictions)
    test_by_dataset = []
    test_rows = by_partition["test"]
    test_labels = labels[test_slice]
    for dataset in sorted({str(row["dataset"]) for row in test_rows}):
        indices = np.asarray(
            [index for index, row in enumerate(test_rows) if row["dataset"] == dataset]
        )
        test_by_dataset.append(
            {
                "dataset": dataset,
                **_evaluation(
                    [test_rows[index] for index in indices],
                    test_labels[indices],
                    test_predictions[indices],
                ),
            }
        )

    trained_at = datetime.now(UTC).isoformat()
    partition_counts = benchmark_report["partition_counts"]
    metadata = {
        "trained_at": trained_at,
        "algorithm": "standardized-logistic-regression",
        "feature_version": FEATURE_VERSION,
        "feature_count": int(features.shape[1]),
        "training_samples": len(by_partition["train"]),
        "validation_samples": len(by_partition["validation"]),
        "test_samples": len(by_partition["test"]),
        "datasets_used_for_training": sorted(
            {str(row["dataset"]) for row in by_partition["train"]}
        ),
        "class_distributions": {partition: partition_counts[partition] for partition in PARTITIONS},
        "split_definitions": benchmark_report["split_definitions"],
        "benchmark_manifest_sha256": benchmark_report["manifest_sha256"],
        "random_state": random_state,
        "decision_threshold": round(float(threshold), 6),
        "fitting_policy": "train-only; validation selects threshold; test remains unseen",
        "independent_benchmark": {
            "schema_version": "1.0",
            "validation": validation,
            "unseen_test": test,
            "unseen_test_by_dataset": test_by_dataset,
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    saved = save_model(
        model_root,
        coefficient=classifier.coef_[0],
        intercept=float(classifier.intercept_[0]),
        feature_mean=scaler.mean_,
        feature_scale=scaler.scale_,
        threshold=float(threshold),
        metadata=metadata,
    )
    return {
        "ready": True,
        "model_name": MODEL_NAME,
        "model_root": str(model_root.resolve()),
        "trained_at": trained_at,
        "training_samples": len(by_partition["train"]),
        "validation": validation,
        "unseen_test": test,
        "unseen_test_by_dataset": test_by_dataset,
        "metadata": saved,
    }
