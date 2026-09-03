from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.ml.features import FEATURE_VERSION

MODEL_SCHEMA_VERSION = "2.0"
MODEL_NAME = "parking-occupancy-logistic-v2"
WEIGHTS_FILENAME = f"{MODEL_NAME}.npz"
METADATA_FILENAME = f"{MODEL_NAME}.json"


class ModelNotReadyError(RuntimeError):
    """Raised when a trained, valid local model is unavailable."""


@dataclass(frozen=True)
class ModelArtifact:
    coefficient: np.ndarray
    intercept: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    threshold: float
    metadata: dict[str, object]


def model_paths(model_root: Path) -> tuple[Path, Path]:
    return model_root / WEIGHTS_FILENAME, model_root / METADATA_FILENAME


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_model(
    model_root: Path,
    *,
    coefficient: np.ndarray,
    intercept: float,
    feature_mean: np.ndarray,
    feature_scale: np.ndarray,
    threshold: float,
    metadata: dict[str, object],
) -> dict[str, object]:
    model_root.mkdir(parents=True, exist_ok=True)
    weights_path, metadata_path = model_paths(model_root)
    with tempfile.NamedTemporaryFile(
        mode="w+b", dir=model_root, prefix=".weights-", suffix=".npz", delete=False
    ) as handle:
        np.savez_compressed(
            handle,
            coefficient=np.asarray(coefficient, dtype=np.float64),
            intercept=np.asarray([intercept], dtype=np.float64),
            feature_mean=np.asarray(feature_mean, dtype=np.float64),
            feature_scale=np.asarray(feature_scale, dtype=np.float64),
            threshold=np.asarray([threshold], dtype=np.float64),
        )
        temporary_weights = Path(handle.name)
    os.replace(temporary_weights, weights_path)

    payload = {
        **metadata,
        "schema_version": MODEL_SCHEMA_VERSION,
        "model_name": MODEL_NAME,
        "feature_version": FEATURE_VERSION,
        "weights_file": WEIGHTS_FILENAME,
        "weights_sha256": _sha256(weights_path),
    }
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=model_root,
        prefix=".metadata-",
        suffix=".json",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary_metadata = Path(handle.name)
    os.replace(temporary_metadata, metadata_path)
    return payload


def load_model(model_root: Path) -> ModelArtifact:
    weights_path, metadata_path = model_paths(model_root)
    if not weights_path.is_file() or not metadata_path.is_file():
        raise ModelNotReadyError("Local model is not trained; run scripts\\train-model.ps1")
    try:
        with metadata_path.open(encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ModelNotReadyError("Model metadata is unreadable") from exc
    if metadata.get("schema_version") != MODEL_SCHEMA_VERSION:
        raise ModelNotReadyError("Unsupported model schema")
    if metadata.get("feature_version") != FEATURE_VERSION:
        raise ModelNotReadyError("Model feature version does not match the application")
    benchmark = metadata.get("independent_benchmark")
    if not isinstance(benchmark, dict) or not isinstance(benchmark.get("unseen_test"), dict):
        raise ModelNotReadyError("Model metadata lacks an independent unseen benchmark")
    if metadata.get("fitting_policy") != (
        "train-only; validation selects threshold; test remains unseen"
    ):
        raise ModelNotReadyError("Model fitting policy is not leakage-safe")
    if metadata.get("weights_sha256") != _sha256(weights_path):
        raise ModelNotReadyError("Model weights checksum does not match metadata")

    try:
        with np.load(weights_path, allow_pickle=False) as arrays:
            coefficient = np.asarray(arrays["coefficient"], dtype=np.float64).reshape(-1)
            intercept = float(np.asarray(arrays["intercept"]).reshape(-1)[0])
            feature_mean = np.asarray(arrays["feature_mean"], dtype=np.float64).reshape(-1)
            feature_scale = np.asarray(arrays["feature_scale"], dtype=np.float64).reshape(-1)
            threshold = float(np.asarray(arrays["threshold"]).reshape(-1)[0])
    except (OSError, ValueError, KeyError, IndexError) as exc:
        raise ModelNotReadyError("Model weights are invalid") from exc
    feature_count = int(metadata.get("feature_count", 0))
    if not feature_count or any(
        array.size != feature_count for array in (coefficient, feature_mean, feature_scale)
    ):
        raise ModelNotReadyError("Model feature dimensions are invalid")
    if not all(np.isfinite(array).all() for array in (coefficient, feature_mean, feature_scale)):
        raise ModelNotReadyError("Model contains non-finite values")
    if not np.isfinite(intercept) or not 0.0 < threshold < 1.0:
        raise ModelNotReadyError("Model scalar values are invalid")
    feature_scale = np.where(feature_scale == 0.0, 1.0, feature_scale)
    return ModelArtifact(
        coefficient=coefficient,
        intercept=intercept,
        feature_mean=feature_mean,
        feature_scale=feature_scale,
        threshold=threshold,
        metadata=metadata,
    )


def model_status(model_root: Path) -> dict[str, object]:
    try:
        artifact = load_model(model_root)
    except ModelNotReadyError as exc:
        return {"ready": False, "model_name": MODEL_NAME, "reason": str(exc)}
    return {
        "ready": True,
        "model_name": MODEL_NAME,
        "feature_version": FEATURE_VERSION,
        "trained_at": artifact.metadata.get("trained_at"),
        "training_samples": artifact.metadata.get("training_samples"),
        "validation_samples": artifact.metadata.get("validation_samples"),
        "test_samples": artifact.metadata.get("test_samples"),
        "independent_benchmark": artifact.metadata.get("independent_benchmark"),
    }
