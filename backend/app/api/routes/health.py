from datetime import UTC, datetime

from fastapi import APIRouter, Response, status
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import engine
from app.ml.model_store import model_status
from app.release import APPLICATION_VERSION
from app.services.reliability_service import collect_readiness

router = APIRouter()


@router.get("/health/live")
def liveness() -> dict[str, object]:
    return {
        "status": "alive",
        "timestamp": datetime.now(UTC).isoformat(),
        "version": APPLICATION_VERSION,
    }


@router.get("/health/ready")
def readiness(response: Response) -> dict[str, object]:
    settings = get_settings()
    report = collect_readiness(settings.parking_data_root, settings.model_root, engine)
    if not report["ready"]:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return report


@router.get("/health")
def health() -> dict[str, object]:
    settings = get_settings()
    database_status = "unavailable"
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        database_status = "connected"
    except Exception:
        database_status = "unavailable"

    return {
        "status": "healthy" if database_status == "connected" else "degraded",
        "version": APPLICATION_VERSION,
        "timestamp": datetime.now(UTC).isoformat(),
        "environment": settings.app_env,
        "database": database_status,
        "dataset_root_exists": settings.parking_data_root.exists(),
        "demo_catalogue_exists": (settings.parking_data_root / "demo" / "catalogue.json").is_file(),
        "model_ready": bool(model_status(settings.model_root)["ready"]),
    }
