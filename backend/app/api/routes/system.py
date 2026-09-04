import platform
import sys

from fastapi import APIRouter

from app.core.config import get_settings
from app.ml.model_store import model_status
from app.release import APPLICATION_VERSION, RELEASE_STAGE, release_manifest

router = APIRouter()


@router.get("")
def system_information() -> dict[str, object]:
    settings = get_settings()
    return {
        "application": settings.app_name,
        "version": APPLICATION_VERSION,
        "release_stage": RELEASE_STAGE,
        "environment": settings.app_env,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": sys.implementation.name,
        "dataset_root": str(settings.parking_data_root),
        "dataset_root_exists": settings.parking_data_root.exists(),
        "live_data_enabled": False,
        "cloud_ai_enabled": False,
        "local_ai_enabled": True,
        "model": model_status(settings.model_root),
    }


@router.get("/release")
def release_information() -> dict[str, object]:
    return release_manifest()
