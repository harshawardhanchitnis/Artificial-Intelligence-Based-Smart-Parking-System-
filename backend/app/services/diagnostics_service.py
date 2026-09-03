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
    return {
        **matrix,
        "evaluated_slots": total,
        "accuracy": _ratio(true_occupied + true_vacant, total),
        "precision": precision,
        "recall": recall,
        "specificity": _ratio(true_vacant, true_vacant + false_occupied),
        "f1_score": round(2 * precision * recall / (precision + recall), 6)
        if precision is not None and recall is not None and precision + recall
        else None,
    }


def model_diagnostics(records: Iterable[AnalysisRecord]) -> dict[str, Any]:
    rows = list(records)
    overall = _empty_matrix()
    by_dataset: dict[str, dict[str, int]] = defaultdict(_empty_matrix)
    eligible_runs = 0
    errors: list[dict[str, object]] = []

    for record in rows:
        try:
            predictions = json.loads(record.prediction_json) if record.prediction_json else []
        except json.JSONDecodeError:
            predictions = []
        if not isinstance(predictions, list) or not predictions:
            continue
        eligible_runs += 1
        for prediction in predictions:
            if not isinstance(prediction, dict):
                continue
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
                        "slot_id": str(prediction.get("id", "")),
                        "predicted_occupied": predicted,
                        "ground_truth_occupied": truth,
                        "confidence": prediction.get("confidence"),
                    }
                )

    return {
        "total_runs": len(rows),
        "eligible_runs": eligible_runs,
        "legacy_runs": len(rows) - eligible_runs,
        "overall": _metrics(overall),
        "dataset_breakdown": [
            {"dataset": dataset, **_metrics(matrix)}
            for dataset, matrix in sorted(by_dataset.items())
        ],
        "recent_errors": errors,
    }
