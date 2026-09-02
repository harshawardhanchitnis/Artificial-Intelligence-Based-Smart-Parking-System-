from __future__ import annotations

import csv
import io

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import AnalysisRecord
from app.db.session import SessionLocal
from app.ml.model_store import model_status
from app.services.analytics_service import analytics_summary, record_payload
from app.services.catalogue_service import CatalogueRepository

router = APIRouter()


def _records() -> list[AnalysisRecord]:
    with SessionLocal() as session:
        return list(
            session.scalars(select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc())).all()
        )


def _record_or_404(analysis_id: int) -> AnalysisRecord:
    with SessionLocal() as session:
        record = session.get(AnalysisRecord, analysis_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Analysis record not found")
        session.expunge(record)
        return record


def _csv_response(rows: list[dict[str, object]], filename: str) -> StreamingResponse:
    columns = [
        "id",
        "dataset",
        "scenario_id",
        "total_spaces",
        "occupied_spaces",
        "vacant_spaces",
        "occupancy_rate",
        "processing_time_ms",
        "model_name",
        "average_confidence",
        "ground_truth_agreement",
        "created_at",
    ]
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/dashboard/summary")
def dashboard_summary() -> dict[str, object]:
    settings = get_settings()
    catalogue = CatalogueRepository(settings.parking_data_root).load()
    records = _records()
    latest = record_payload(records[0]) if records else None
    return {
        "prepared": bool(catalogue.get("prepared")),
        "scenario_count": int(catalogue.get("scenario_count", 0)),
        "datasets": catalogue.get("datasets", []),
        "model": model_status(settings.model_root),
        "analysis_count": len(records),
        "latest_analysis": latest,
    }


@router.get("/analytics/summary")
def summary() -> dict[str, object]:
    return analytics_summary(_records())


@router.get("/reports/analyses.csv")
def analyses_csv() -> StreamingResponse:
    return _csv_response(
        [record_payload(record) for record in _records()],
        "smart-parking-analysis-history.csv",
    )


@router.get("/reports/analyses/{analysis_id}.json")
def analysis_json(analysis_id: int) -> JSONResponse:
    payload = record_payload(_record_or_404(analysis_id), include_predictions=True)
    return JSONResponse(
        content=_json_safe(payload),
        headers={
            "Content-Disposition": (
                f'attachment; filename="smart-parking-analysis-{analysis_id}.json"'
            )
        },
    )


@router.get("/reports/analyses/{analysis_id}.csv")
def analysis_csv(analysis_id: int) -> StreamingResponse:
    return _csv_response(
        [record_payload(_record_or_404(analysis_id))],
        f"smart-parking-analysis-{analysis_id}.csv",
    )


def _json_safe(payload: dict[str, object]) -> dict[str, object]:
    created_at = payload.get("created_at")
    if hasattr(created_at, "isoformat"):
        payload["created_at"] = created_at.isoformat()
    return payload
