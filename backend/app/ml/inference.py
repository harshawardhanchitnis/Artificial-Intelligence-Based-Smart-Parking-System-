from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image

from app.ml.features import scenario_features
from app.ml.geometry import rectify_slot
from app.ml.model_store import load_model
from app.ml.occupancy_v3 import OccupancyV3NotReadyError, OccupancyV3Predictor
from app.services.catalogue_service import CatalogueRepository


def analyse_scenario(data_root: Path, model_root: Path, scenario_id: str) -> dict[str, object]:
    started = perf_counter()
    repository = CatalogueRepository(data_root)
    scenario = repository.get_scenario(scenario_id)
    slots = scenario.get("slots")
    if not isinstance(slots, list) or not slots:
        raise ValueError("Scenario contains no parking spaces")
    with Image.open(repository.image_path(scenario_id)) as image:
        rgb = image.convert("RGB")
        try:
            enhanced = OccupancyV3Predictor.load(model_root)
            occupied_probabilities, _ = enhanced.probabilities(
                [rectify_slot(rgb, slot["polygon"]) for slot in slots]
            )
            threshold = enhanced.threshold
            model_name = str(enhanced.metadata["model_name"])
            feature_version = str(enhanced.metadata["preprocessing"])
            trained_at = enhanced.metadata.get("trained_at")
        except OccupancyV3NotReadyError:
            artifact = load_model(model_root)
            features = scenario_features(rgb, slots).astype(np.float64)
            normalized = (features - artifact.feature_mean) / artifact.feature_scale
            logits = np.clip(normalized @ artifact.coefficient + artifact.intercept, -40, 40)
            occupied_probabilities = 1.0 / (1.0 + np.exp(-logits))
            threshold = artifact.threshold
            model_name = str(artifact.metadata["model_name"])
            feature_version = str(artifact.metadata["feature_version"])
            trained_at = artifact.metadata.get("trained_at")
    predicted_states = occupied_probabilities >= threshold
    predictions = []
    agreements = 0
    for slot, probability, predicted in zip(
        slots, occupied_probabilities, predicted_states, strict=True
    ):
        truth = bool(slot["occupied"])
        prediction = bool(predicted)
        agreements += int(truth == prediction)
        confidence = max(float(probability), 1.0 - float(probability))
        predictions.append(
            {
                "id": str(slot["id"]),
                "polygon": slot["polygon"],
                "predicted_occupied": prediction,
                "occupied_probability": round(float(probability), 6),
                "confidence": round(confidence, 6),
                "ground_truth_occupied": truth,
                "correct": truth == prediction,
            }
        )
    occupied = int(np.sum(predicted_states))
    total = len(predictions)
    elapsed_ms = (perf_counter() - started) * 1_000
    return {
        "scenario_id": scenario_id,
        "dataset": scenario["dataset"],
        "lot": scenario["lot"],
        "condition": scenario["condition"],
        "model_name": model_name,
        "feature_version": feature_version,
        "trained_at": trained_at,
        "decision_threshold": threshold,
        "total_spaces": total,
        "occupied_spaces": occupied,
        "vacant_spaces": total - occupied,
        "ground_truth_agreement": round(agreements / total, 6),
        "average_confidence": round(
            float(np.mean([prediction["confidence"] for prediction in predictions])), 6
        ),
        "processing_time_ms": round(elapsed_ms, 3),
        "predictions": predictions,
    }


def verify_model(data_root: Path, model_root: Path) -> dict[str, object]:
    repository = CatalogueRepository(data_root)
    scenarios = repository.list_scenarios(limit=10_000)
    if not scenarios:
        raise ValueError("Prepared catalogue contains no scenarios")
    results = [
        analyse_scenario(data_root, model_root, str(scenario["id"])) for scenario in scenarios
    ]
    total_slots = sum(int(result["total_spaces"]) for result in results)
    weighted_agreement = (
        sum(
            float(result["ground_truth_agreement"]) * int(result["total_spaces"])
            for result in results
        )
        / total_slots
    )
    return {
        "valid": True,
        "scenario_count": len(results),
        "slot_count": total_slots,
        "catalogue_smoke_agreement": round(weighted_agreement, 6),
        "average_processing_time_ms": round(
            sum(float(result["processing_time_ms"]) for result in results) / len(results),
            3,
        ),
        "datasets": sorted({str(result["dataset"]) for result in results}),
    }
