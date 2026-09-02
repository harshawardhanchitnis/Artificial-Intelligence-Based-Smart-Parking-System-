from datetime import UTC, datetime

from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import engine

router = APIRouter()


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
        "timestamp": datetime.now(UTC).isoformat(),
        "environment": settings.app_env,
        "database": database_status,
        "dataset_root_exists": settings.parking_data_root.exists(),
    }
