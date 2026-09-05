from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Query
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import AnalysisRecord
from app.db.session import SessionLocal
from app.ml.inference import analyse_scenario
from app.ml.localization import localizer_status
from app.ml.model_store import ModelNotReadyError, model_status
from app.ml.occupancy_v3 import occupancy_v3_status
from app.services.analytics_service import record_payload
from app.services.catalogue_service import ScenarioNotFoundError

router = APIRouter()


@router.get("/model/status")
def status() -> dict[str, object]:
    settings = get_settings()
    model_bundle = model_status(settings.model_root)
    baseline = model_bundle["baseline"]
    enhanced = occupancy_v3_status(settings.model_root)
    active = enhanced if enhanced["ready"] else baseline
    return {
        **active,
        "ready": bool(enhanced["ready"] or baseline["ready"]),
        "active_model": enhanced if enhanced["ready"] else baseline,
        "enhanced_occupancy": enhanced,
        "automatic_localizer": localizer_status(settings.model_root),
        "reproducible_baseline": baseline,
    }


@router.post("/analysis/scenarios/{scenario_id}")
def run_analysis(scenario_id: str) -> dict[str, object]:
    settings = get_settings()
    try:
        result = analyse_scenario(settings.parking_data_root, settings.model_root, scenario_id)
    except ScenarioNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Scenario not found") from exc
    except ModelNotReadyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    record = AnalysisRecord(
        dataset=str(result["dataset"]),
        scenario_id=scenario_id,
        total_spaces=int(result["total_spaces"]),
        occupied_spaces=int(result["occupied_spaces"]),
        vacant_spaces=int(result["vacant_spaces"]),
        processing_time_ms=float(result["processing_time_ms"]),
        result_image_path=None,
        model_name=str(result["model_name"]),
        average_confidence=float(result["average_confidence"]),
        ground_truth_agreement=float(result["ground_truth_agreement"]),
        prediction_json=json.dumps(result["predictions"], separators=(",", ":")),
    )
    with SessionLocal() as session:
        session.add(record)
        session.commit()
        session.refresh(record)
    return {**result, "analysis_id": record.id, "created_at": record.created_at}


@router.get("/analysis/history")
def history(limit: int = Query(default=25, ge=1, le=200)) -> dict[str, object]:
    with SessionLocal() as session:
        records = session.scalars(
            select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc()).limit(limit)
        ).all()
    rows = [record_payload(record) for record in records]
    return {"count": len(rows), "analyses": rows}


@router.get("/analysis/history/{analysis_id}")
def history_detail(analysis_id: int) -> dict[str, object]:
    with SessionLocal() as session:
        record = session.get(AnalysisRecord, analysis_id)
        if record is None:
            raise HTTPException(status_code=404, detail="Analysis record not found")
        payload = record_payload(record, include_predictions=True)
    return payload
