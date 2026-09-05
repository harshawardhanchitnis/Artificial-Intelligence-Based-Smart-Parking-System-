from fastapi import APIRouter
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import AnalysisRecord
from app.db.session import SessionLocal
from app.ml.model_store import ModelNotReadyError, load_model
from app.ml.occupancy_v3 import occupancy_v3_status
from app.services.diagnostics_service import diagnostics_report

router = APIRouter()


@router.get("/summary")
def summary() -> dict[str, object]:
    with SessionLocal() as session:
        records = session.scalars(
            select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc())
        ).all()
    try:
        benchmark = load_model(get_settings().model_root).metadata.get("independent_benchmark")
    except ModelNotReadyError:
        benchmark = None
    report = diagnostics_report(records, benchmark if isinstance(benchmark, dict) else None)
    report["enhanced_occupancy"] = occupancy_v3_status(get_settings().model_root)
    return report
