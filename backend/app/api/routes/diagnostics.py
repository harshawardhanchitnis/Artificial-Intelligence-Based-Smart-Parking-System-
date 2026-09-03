from fastapi import APIRouter
from sqlalchemy import select

from app.db.models import AnalysisRecord
from app.db.session import SessionLocal
from app.services.diagnostics_service import model_diagnostics

router = APIRouter()


@router.get("/summary")
def summary() -> dict[str, object]:
    with SessionLocal() as session:
        records = session.scalars(
            select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc())
        ).all()
    return model_diagnostics(records)
