from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from app.db.models import AnalysisRecord

# Datasets with verified ground truth.  Anything else is a user-supplied run
# and must not be presented beside them as though it were a benchmark.
BENCHMARK_DATASETS = frozenset({"PKLot", "CNRPark+EXT", "ACPDS"})


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 6) if values else None


def record_payload(
    record: AnalysisRecord,
    *,
    include_predictions: bool = False,
    media_names: dict[int, str] | None = None,
) -> dict[str, Any]:
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
        "source_type": record.source_type,
        "media_asset_id": record.media_asset_id,
        "job_id": record.job_id,
        "layout_id": record.layout_id,
        "localization_confidence": record.localization_confidence,
        "result_status": record.result_status,
        # The filename the operator recognises.  Falls back to the scenario id,
        # which for an upload is only a content hash.
        "display_name": (
            (media_names or {}).get(record.media_asset_id or -1) or record.scenario_id
        ),
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

    # Image analysis and whole-video jobs differ by more than an order of
    # magnitude, so a single mean across both describes neither.  They are
    # reported separately and never blended.
    image_rows = [row for row in rows if row.source_type != "video_upload"]
    video_rows = [row for row in rows if row.source_type == "video_upload"]

    agreement_rows = [row for row in rows if row.ground_truth_agreement is not None]

    return {
        "total_runs": len(rows),
        "total_spaces_analysed": sum(record.total_spaces for record in rows),
        "average_image_analysis_ms": _mean(
            [float(record.processing_time_ms) for record in image_rows]
        ),
        "average_video_job_ms": _mean([float(record.processing_time_ms) for record in video_rows]),
        "image_runs": len(image_rows),
        "video_runs": len(video_rows),
        "average_confidence": _mean(
            [
                float(record.average_confidence)
                for record in rows
                if record.average_confidence is not None
            ]
        ),
        "ground_truth_agreement": _mean(
            [float(row.ground_truth_agreement) for row in agreement_rows]
        ),
        # The agreement figure only covers runs that have ground truth; naming
        # the count stops it being read as a whole-history accuracy.
        "ground_truth_runs": len(agreement_rows),
        "dataset_breakdown": [
            row for row in dataset_breakdown if row["dataset"] in BENCHMARK_DATASETS
        ],
        "user_run_breakdown": [
            row for row in dataset_breakdown if row["dataset"] not in BENCHMARK_DATASETS
        ],
        "recent_runs": [record_payload(record) for record in reversed(rows[:10])],
    }
