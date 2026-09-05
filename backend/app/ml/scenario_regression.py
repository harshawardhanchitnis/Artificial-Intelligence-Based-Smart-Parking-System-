from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from app.datasets.integrity import atomic_json, sha256_file
from app.ml.features import scenario_features
from app.ml.geometry import rectify_slot, validate_polygon
from app.ml.model_store import load_model
from app.ml.occupancy_v3 import OccupancyV3Predictor
from app.ml.training import classification_metrics
from app.services.catalogue_service import CatalogueRepository


def _error_summary(
    truth: np.ndarray, probabilities: np.ndarray, threshold: float
) -> dict[str, object]:
    predictions = probabilities >= threshold
    wrong = predictions != truth
    confidence = np.maximum(probabilities, 1.0 - probabilities)
    return {
        **classification_metrics(truth.astype(np.int64), predictions.astype(np.int64)),
        "false_vacant": int(np.sum((truth == 1) & ~predictions)),
        "false_occupied": int(np.sum((truth == 0) & predictions)),
        "high_confidence_wrong": int(np.sum(wrong & (confidence >= 0.8))),
        "low_confidence_borderline_wrong": int(
            np.sum(wrong & (np.abs(probabilities - threshold) <= 0.1))
        ),
        "mean_confidence": round(float(np.mean(confidence)), 6),
        "samples": int(len(truth)),
    }


def audit_prepared_scenarios(
    data_root: Path, model_root: Path, *, output_path: Path | None = None
) -> dict[str, object]:
    """Compare frozen V2 and enhanced V3 on exposed scenarios without selecting a model."""
    repository = CatalogueRepository(data_root)
    scenarios = repository.list_scenarios(limit=10_000)
    baseline = load_model(model_root)
    enhanced = OccupancyV3Predictor.load(model_root)
    rows: list[dict[str, object]] = []
    aggregate: dict[str, dict[str, list[float]]] = {
        "baseline": defaultdict(list),
        "enhanced": defaultdict(list),
    }
    dataset_aggregate: dict[tuple[str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    geometry_invalid = 0
    started = perf_counter()
    for scenario in scenarios:
        slots = scenario["slots"]
        geometry_invalid += sum(
            not validate_polygon(slot["polygon"]).valid
            for slot in slots  # type: ignore[index]
        )
        with Image.open(repository.image_path(str(scenario["id"]))) as source:
            image = source.convert("RGB")
            old_features = scenario_features(image, slots).astype(np.float64)  # type: ignore[arg-type]
            old_logits = np.clip(
                ((old_features - baseline.feature_mean) / baseline.feature_scale)
                @ baseline.coefficient
                + baseline.intercept,
                -40,
                40,
            )
            old_probabilities = 1.0 / (1.0 + np.exp(-old_logits))
            new_probabilities, _ = enhanced.probabilities(
                [rectify_slot(image, slot["polygon"]) for slot in slots]  # type: ignore[index]
            )
        truth = np.asarray([int(bool(slot["occupied"])) for slot in slots])  # type: ignore[index]
        old = _error_summary(truth, old_probabilities, baseline.threshold)
        new = _error_summary(truth, new_probabilities, enhanced.threshold)
        row = {
            "scenario_id": scenario["id"],
            "dataset": scenario["dataset"],
            "lot": scenario["lot"],
            "condition": scenario["condition"],
            "slot_count": len(slots),  # type: ignore[arg-type]
            "baseline": old,
            "enhanced": new,
            "accuracy_delta": round(float(new["accuracy"]) - float(old["accuracy"]), 6),
        }
        rows.append(row)
        for name, probabilities, threshold in (
            ("baseline", old_probabilities, baseline.threshold),
            ("enhanced", new_probabilities, enhanced.threshold),
        ):
            aggregate[name]["truth"].extend(truth.tolist())
            aggregate[name]["probabilities"].extend(probabilities.tolist())
            aggregate[name]["threshold"].append(float(threshold))
            dataset_values = dataset_aggregate[(str(scenario["dataset"]), name)]
            dataset_values["truth"].extend(truth.tolist())
            dataset_values["probabilities"].extend(probabilities.tolist())
            dataset_values["threshold"].append(float(threshold))

    overall = {}
    for name, values in aggregate.items():
        overall[name] = _error_summary(
            np.asarray(values["truth"]),
            np.asarray(values["probabilities"]),
            float(values["threshold"][0]),
        )
    report = {
        "purpose": "exposed prepared-scenario regression; never used for model selection",
        "catalogue_sha256": sha256_file(data_root / "demo" / "catalogue.json"),
        "model_artifacts": {
            "baseline_weights_sha256": sha256_file(
                model_root / "parking-occupancy-logistic-v2.npz"
            ),
            "enhanced_onnx_sha256": str(enhanced.metadata["onnx_sha256"]),
        },
        "catalogue_scenarios": len(rows),
        "geometry": {
            "invalid_polygons": geometry_invalid,
            "baseline_preprocessing": "padded axis-aligned bbox resized to 32x32",
            "enhanced_preprocessing": (
                "ordered polygon perspective rectification and mask at 128x128"
            ),
        },
        "overall": overall,
        "by_dataset": {
            dataset: {
                name: _error_summary(
                    np.asarray(dataset_aggregate[(dataset, name)]["truth"]),
                    np.asarray(dataset_aggregate[(dataset, name)]["probabilities"]),
                    float(dataset_aggregate[(dataset, name)]["threshold"][0]),
                )
                for name in ("baseline", "enhanced")
            }
            for dataset in sorted({str(scenario["dataset"]) for scenario in scenarios})
        },
        "scenarios": sorted(
            rows,
            key=lambda row: (
                float(row["enhanced"]["accuracy"]),  # type: ignore[index]
                str(row["dataset"]),
                str(row["scenario_id"]),
            ),
        ),
        "elapsed_ms": round((perf_counter() - started) * 1_000, 3),
        "interpretation": {
            "roi_layout": (
                "Invalid polygon count isolates annotation/layout defects; zero invalid polygons "
                "does not prove perfect alignment, so scenario QC overlays remain mandatory."
            ),
            "domain": (
                "Per-condition and per-site rows expose weather, camera and dataset shift. "
                "High-confidence errors indicate representation/domain failures; borderline errors "
                "are candidates for calibration analysis."
            ),
        },
    }
    if output_path:
        atomic_json(output_path, report)
    return report
