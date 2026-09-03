from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from app.db.models import AnalysisRecord


def _ratio(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def _empty_matrix() -> dict[str, int]:
    return {"true_occupied": 0, "true_vacant": 0, "false_occupied": 0, "false_vacant": 0}


def _metrics(matrix: dict[str, int]) -> dict[str, float | int | None]:
    true_occupied = matrix["true_occupied"]
    true_vacant = matrix["true_vacant"]
    false_occupied = matrix["false_occupied"]
    false_vacant = matrix["false_vacant"]
    total = sum(matrix.values())
    precision = _ratio(true_occupied, true_occupied + false_occupied)
    recall = _ratio(true_occupied, true_occupied + false_vacant)
    specificity = _ratio(true_vacant, true_vacant + false_occupied)
    balanced_accuracy = (
        round((recall + specificity) / 2, 6)
        if recall is not None and specificity is not None
        else None
    )
    return {
        **matrix,
        "evaluated_slots": total,
        "accuracy": _ratio(true_occupied + true_vacant, total),
        "balanced_accuracy": balanced_accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1_score": round(2 * precision * recall / (precision + recall), 6)
        if precision is not None and recall is not None and precision + recall
        else None,
    }


def application_run_diagnostics(records: Iterable[AnalysisRecord]) -> dict[str, Any]:
    rows = list(records)
    overall = _empty_matrix()
    by_dataset: dict[str, dict[str, int]] = defaultdict(_empty_matrix)
    eligible_runs = 0
    errors: list[dict[str, object]] = []
    unique_scenarios: set[tuple[str, str]] = set()
    unique_slots: set[tuple[str, str, str]] = set()

    for record in rows:
        try:
            predictions = json.loads(record.prediction_json) if record.prediction_json else []
        except json.JSONDecodeError:
            predictions = []
        if not isinstance(predictions, list) or not predictions:
            continue
        eligible_runs += 1
        unique_scenarios.add((record.dataset, record.scenario_id))
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
            slot_id = str(prediction.get("id", ""))
            unique_slots.add((record.dataset, record.scenario_id, slot_id))
            predicted = bool(prediction.get("predicted_occupied"))
            truth = bool(prediction.get("ground_truth_occupied"))
            if predicted and truth:
                outcome = "true_occupied"
            elif not predicted and not truth:
                outcome = "true_vacant"
            elif predicted:
                outcome = "false_occupied"
            else:
                outcome = "false_vacant"
            overall[outcome] += 1
            by_dataset[record.dataset][outcome] += 1
            if predicted != truth and len(errors) < 25:
                errors.append(
                    {
                        "analysis_id": record.id,
                        "dataset": record.dataset,
                        "scenario_id": record.scenario_id,
                        "slot_id": slot_id,
                        "predicted_occupied": predicted,
                        "ground_truth_occupied": truth,
                        "confidence": prediction.get("confidence"),
                    }
                )

    metrics = _metrics(overall)
    repeated = int(metrics["evaluated_slots"]) - len(unique_slots)
    return {
        "semantic_label": "Stored-run agreement",
        "total_runs": len(rows),
        "prediction_enabled_runs": eligible_runs,
        "legacy_runs": len(rows) - eligible_runs,
        "stored_slot_predictions": metrics["evaluated_slots"],
        "unique_scenarios": len(unique_scenarios),
        "unique_scenario_slots": len(unique_slots),
        "repeated_slot_predictions": repeated,
        "includes_repeated_executions": repeated > 0,
        "context_note": (
            "Repeated executions are included. These figures describe saved application runs "
            "and are not an independent estimate of model generalization."
        ),
        "overall": metrics,
        "dataset_breakdown": [
            {"dataset": dataset, **_metrics(matrix)}
            for dataset, matrix in sorted(by_dataset.items())
        ],
        "recent_errors": errors,
    }


def diagnostics_report(
    records: Iterable[AnalysisRecord], independent_benchmark: dict[str, object] | None
) -> dict[str, Any]:
    return {
        "application_runs": application_run_diagnostics(records),
        "independent_benchmark": independent_benchmark,
        "benchmark_available": independent_benchmark is not None,
    }


def model_diagnostics(records: Iterable[AnalysisRecord]) -> dict[str, Any]:
    """Backward-compatible service alias used by older integrations."""
    return application_run_diagnostics(records)
