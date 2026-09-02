from datetime import UTC, datetime

from app.db.models import AnalysisRecord
from app.services.analytics_service import analytics_summary, record_payload


def record(dataset: str, occupied: int, confidence: float | None) -> AnalysisRecord:
    return AnalysisRecord(
        id=1,
        dataset=dataset,
        scenario_id="fixture",
        total_spaces=10,
        occupied_spaces=occupied,
        vacant_spaces=10 - occupied,
        processing_time_ms=20.0,
        model_name="fixture-model",
        average_confidence=confidence,
        ground_truth_agreement=0.9 if confidence is not None else None,
        prediction_json='[{"id":"1","correct":true}]',
        created_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


def test_analytics_summary_handles_current_and_legacy_rows() -> None:
    summary = analytics_summary([record("PKLot", 6, 0.8), record("ACPDS", 4, None)])

    assert summary["total_runs"] == 2
    assert summary["total_spaces_analysed"] == 20
    assert summary["average_confidence"] == 0.8
    assert len(summary["dataset_breakdown"]) == 2


def test_detailed_report_includes_saved_predictions() -> None:
    payload = record_payload(record("PKLot", 6, 0.8), include_predictions=True)

    assert payload["occupancy_rate"] == 0.6
    assert payload["predictions"][0]["correct"] is True
