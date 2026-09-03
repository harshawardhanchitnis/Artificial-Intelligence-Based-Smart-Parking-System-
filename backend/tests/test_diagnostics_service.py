import json
from datetime import UTC, datetime
from types import SimpleNamespace

from app.api.routes import diagnostics as diagnostics_route
from app.db.models import AnalysisRecord
from app.services.diagnostics_service import (
    application_run_diagnostics,
    diagnostics_report,
)


def analysis(
    dataset: str,
    predictions: list[dict[str, object]] | None,
    *,
    record_id: int = 1,
    scenario_id: str = "fixture",
) -> AnalysisRecord:
    return AnalysisRecord(
        id=record_id,
        dataset=dataset,
        scenario_id=scenario_id,
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
    report = application_run_diagnostics([analysis("PKLot", predictions), analysis("ACPDS", None)])
    assert report["prediction_enabled_runs"] == 1
    assert report["legacy_runs"] == 1
    assert report["overall"]["evaluated_slots"] == 4
    assert report["overall"]["accuracy"] == 0.5
    assert report["overall"]["balanced_accuracy"] == 0.5
    assert len(report["recent_errors"]) == 2


def test_repeated_runs_are_explicit_and_do_not_change_benchmark() -> None:
    predictions = [
        {"id": "1", "predicted_occupied": False, "ground_truth_occupied": False},
        {"id": "2", "predicted_occupied": True, "ground_truth_occupied": True},
    ]
    benchmark = {"unseen_test": {"unique_samples": 600}}
    report = diagnostics_report(
        [
            analysis("PKLot", predictions, record_id=1),
            analysis("PKLot", predictions, record_id=2),
        ],
        benchmark,
    )
    runs = report["application_runs"]
    assert runs["stored_slot_predictions"] == 4
    assert runs["unique_scenario_slots"] == 2
    assert runs["repeated_slot_predictions"] == 2
    assert runs["includes_repeated_executions"] is True
    assert report["independent_benchmark"]["unseen_test"]["unique_samples"] == 600


def test_diagnostics_handles_no_prediction_enabled_runs() -> None:
    report = application_run_diagnostics([analysis("PKLot", None)])
    assert report["overall"]["evaluated_slots"] == 0
    assert report["overall"]["accuracy"] is None
    assert report["dataset_breakdown"] == []


def test_diagnostics_api_separates_benchmark_from_application_runs(monkeypatch) -> None:
    benchmark = {"unseen_test": {"unique_samples": 1800}}

    class Result:
        def all(self):
            return []

    class Session:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def scalars(self, _query):
            return Result()

    monkeypatch.setattr(diagnostics_route, "SessionLocal", Session)
    monkeypatch.setattr(
        diagnostics_route,
        "load_model",
        lambda _root: SimpleNamespace(metadata={"independent_benchmark": benchmark}),
    )
    response = diagnostics_route.summary()
    assert response["benchmark_available"] is True
    assert response["independent_benchmark"] == benchmark
    assert response["application_runs"]["semantic_label"] == "Stored-run agreement"
