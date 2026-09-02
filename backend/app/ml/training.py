from __future__ import annotations

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
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

from app.ml.features import FEATURE_VERSION, scenario_features
from app.ml.model_store import MODEL_NAME, save_model
from app.services.catalogue_service import CatalogueRepository


class TrainingError(RuntimeError):
    """Raised when a valid occupancy model cannot be trained."""


def _metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, object]:
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(labels, predictions)), 6),
        "balanced_accuracy": round(float(balanced_accuracy_score(labels, predictions)), 6),
        "precision_occupied": round(
            float(precision_score(labels, predictions, zero_division=0)), 6
        ),
        "recall_occupied": round(float(recall_score(labels, predictions, zero_division=0)), 6),
        "f1_occupied": round(float(f1_score(labels, predictions, zero_division=0)), 6),
        "confusion_matrix": matrix.tolist(),
    }


def _choose_group_split(
    features: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    *,
    validation_fraction: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray]:
    splitter = GroupShuffleSplit(
        n_splits=40, test_size=validation_fraction, random_state=random_state
    )
    for train_indices, validation_indices in splitter.split(features, labels, groups):
        if (
            np.unique(labels[train_indices]).size == 2
            and np.unique(labels[validation_indices]).size == 2
        ):
            return train_indices, validation_indices
    raise TrainingError("Could not produce a group holdout containing both classes")


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


def train_model(
    data_root: Path,
    model_root: Path,
    *,
    validation_fraction: float = 0.25,
    random_state: int = 42,
    progress=print,
) -> dict[str, object]:
    if not 0.15 <= validation_fraction <= 0.4:
        raise TrainingError("validation_fraction must be between 0.15 and 0.40")
    repository = CatalogueRepository(data_root)
    scenarios = repository.list_scenarios(limit=10_000)
    if len(scenarios) < 3:
        raise TrainingError("At least three prepared scenarios are required")

    feature_rows: list[np.ndarray] = []
    labels: list[int] = []
    groups: list[str] = []
    dataset_counts: dict[str, int] = {}
    for index, scenario in enumerate(scenarios, start=1):
        scenario_id = str(scenario["id"])
        image_path = repository.image_path(scenario_id)
        slots = scenario.get("slots")
        if not isinstance(slots, list) or not slots:
            continue
        with Image.open(image_path) as image:
            extracted = scenario_features(image.convert("RGB"), slots)
        feature_rows.extend(extracted)
        labels.extend(int(bool(slot["occupied"])) for slot in slots)
        groups.extend([scenario_id] * len(slots))
        dataset = str(scenario["dataset"])
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + len(slots)
        if progress:
            progress(f"Features: {index}/{len(scenarios)} scenarios ({len(labels)} slots)")

    features = np.asarray(feature_rows, dtype=np.float64)
    label_array = np.asarray(labels, dtype=np.int64)
    group_array = np.asarray(groups)
    if features.ndim != 2 or features.shape[0] < 20:
        raise TrainingError("At least 20 labelled slots are required")
    if np.unique(label_array).size != 2:
        raise TrainingError("Training data must include vacant and occupied spaces")

    train_indices, validation_indices = _choose_group_split(
        features,
        label_array,
        group_array,
        validation_fraction=validation_fraction,
        random_state=random_state,
    )
    validation_scaler, validation_model = _fit(
        features[train_indices], label_array[train_indices], random_state
    )
    probabilities = validation_model.predict_proba(
        validation_scaler.transform(features[validation_indices])
    )[:, 1]
    threshold_candidates = np.linspace(0.3, 0.7, 17)
    threshold = max(
        threshold_candidates,
        key=lambda value: balanced_accuracy_score(
            label_array[validation_indices], probabilities >= value
        ),
    )
    validation_predictions = (probabilities >= threshold).astype(np.int64)
    validation_metrics = _metrics(label_array[validation_indices], validation_predictions)

    final_scaler, final_model = _fit(features, label_array, random_state)
    trained_at = datetime.now(UTC).isoformat()
    metadata = {
        "trained_at": trained_at,
        "algorithm": "standardized-logistic-regression",
        "feature_version": FEATURE_VERSION,
        "feature_count": int(features.shape[1]),
        "training_samples": int(features.shape[0]),
        "scenario_count": len(set(groups)),
        "class_counts": {
            "vacant": int(np.sum(label_array == 0)),
            "occupied": int(np.sum(label_array == 1)),
        },
        "dataset_slot_counts": dict(sorted(dataset_counts.items())),
        "validation_strategy": "scenario-group-holdout",
        "validation_samples": int(validation_indices.size),
        "validation_scenarios": sorted(set(group_array[validation_indices].tolist())),
        "validation_metrics": validation_metrics,
        "decision_threshold": round(float(threshold), 6),
        "random_state": random_state,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    saved = save_model(
        model_root,
        coefficient=final_model.coef_[0],
        intercept=float(final_model.intercept_[0]),
        feature_mean=final_scaler.mean_,
        feature_scale=final_scaler.scale_,
        threshold=float(threshold),
        metadata=metadata,
    )
    return {
        "ready": True,
        "model_name": MODEL_NAME,
        "model_root": str(model_root.resolve()),
        "trained_at": trained_at,
        "training_samples": int(features.shape[0]),
        "validation_metrics": validation_metrics,
        "metadata": saved,
    }
