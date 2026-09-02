from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from app.db.models import AnalysisRecord


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def record_payload(record: AnalysisRecord, *, include_predictions: bool = False) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "id": record.id,
        "dataset": record.dataset,
        "scenario_id": record.scenario_id,
        "total_spaces": record.total_spaces,
        "occupied_spaces": record.occupied_spaces,
        "vacant_spaces": record.vacant_spaces,
        "occupancy_rate": round(record.occupied_spaces / record.total_spaces, 6)
        if record.total_spaces
        else 0.0,
        "processing_time_ms": record.processing_time_ms,
        "model_name": record.model_name,
        "average_confidence": record.average_confidence,
        "ground_truth_agreement": record.ground_truth_agreement,
        "created_at": record.created_at,
    }
    if include_predictions:
        try:
            predictions = json.loads(record.prediction_json) if record.prediction_json else []
        except json.JSONDecodeError:
            predictions = []
        payload["predictions"] = predictions if isinstance(predictions, list) else []
    return payload


def analytics_summary(records: Iterable[AnalysisRecord]) -> dict[str, Any]:
    rows = list(records)
    groups: dict[str, list[AnalysisRecord]] = defaultdict(list)
    for record in rows:
        groups[record.dataset].append(record)

    dataset_breakdown = []
    for dataset, group in sorted(groups.items()):
        spaces = sum(record.total_spaces for record in group)
        occupied = sum(record.occupied_spaces for record in group)
        dataset_breakdown.append(
            {
                "dataset": dataset,
                "runs": len(group),
                "spaces_analysed": spaces,
                "occupancy_rate": round(occupied / spaces, 6) if spaces else 0.0,
                "average_confidence": _mean(
                    [
                        float(row.average_confidence)
                        for row in group
                        if row.average_confidence is not None
                    ]
                ),
                "ground_truth_agreement": _mean(
                    [
                        float(row.ground_truth_agreement)
                        for row in group
                        if row.ground_truth_agreement is not None
                    ]
                ),
            }
        )

    return {
        "total_runs": len(rows),
        "total_spaces_analysed": sum(record.total_spaces for record in rows),
        "average_processing_time_ms": _mean([float(record.processing_time_ms) for record in rows]),
        "average_confidence": _mean(
            [
                float(record.average_confidence)
                for record in rows
                if record.average_confidence is not None
            ]
        ),
        "ground_truth_agreement": _mean(
            [
                float(record.ground_truth_agreement)
                for record in rows
                if record.ground_truth_agreement is not None
            ]
        ),
        "dataset_breakdown": dataset_breakdown,
        "recent_runs": [record_payload(record) for record in reversed(rows[:10])],
    }
