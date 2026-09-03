import json
from datetime import UTC, datetime

from app.db.models import AnalysisRecord
from app.services.diagnostics_service import model_diagnostics


def analysis(dataset: str, predictions: list[dict[str, object]] | None) -> AnalysisRecord:
    return AnalysisRecord(
        id=1,
        dataset=dataset,
        scenario_id="fixture",
        total_spaces=len(predictions or []),
        occupied_spaces=0,
        vacant_spaces=len(predictions or []),
        processing_time_ms=1.0,
        prediction_json=json.dumps(predictions) if predictions is not None else None,
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_diagnostics_builds_confusion_matrix_and_metrics() -> None:
    predictions = [
        {"id": "tp", "predicted_occupied": True, "ground_truth_occupied": True},
        {"id": "tn", "predicted_occupied": False, "ground_truth_occupied": False},
        {"id": "fp", "predicted_occupied": True, "ground_truth_occupied": False},
        {"id": "fn", "predicted_occupied": False, "ground_truth_occupied": True},
    ]

    report = model_diagnostics([analysis("PKLot", predictions), analysis("ACPDS", None)])

    assert report["eligible_runs"] == 1
    assert report["legacy_runs"] == 1
    assert report["overall"]["evaluated_slots"] == 4
    assert report["overall"]["accuracy"] == 0.5
    assert report["overall"]["precision"] == 0.5
    assert report["overall"]["recall"] == 0.5
    assert report["overall"]["specificity"] == 0.5
    assert report["overall"]["f1_score"] == 0.5
    assert len(report["recent_errors"]) == 2


def test_diagnostics_handles_no_prediction_enabled_runs() -> None:
    report = model_diagnostics([analysis("PKLot", None)])

    assert report["overall"]["evaluated_slots"] == 0
    assert report["overall"]["accuracy"] is None
    assert report["dataset_breakdown"] == []
