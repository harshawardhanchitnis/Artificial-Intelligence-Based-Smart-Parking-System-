import platform
import sys

from fastapi import APIRouter

from app.core.config import get_settings

router = APIRouter()


@router.get("")
def system_information() -> dict[str, object]:
    settings = get_settings()
    return {
        "application": settings.app_name,
        "environment": settings.app_env,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "implementation": sys.implementation.name,
        "dataset_root": str(settings.parking_data_root),
        "dataset_root_exists": settings.parking_data_root.exists(),
        "live_data_enabled": False,
        "cloud_ai_enabled": False,
    }
