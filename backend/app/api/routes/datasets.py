from fastapi import APIRouter, Query

from app.core.config import get_settings
from app.datasets.archive_validator import validate_archive_catalogue

router = APIRouter()


@router.get("/archives")
def archives(deep: bool = Query(default=False)) -> dict[str, object]:
    settings = get_settings()
    results = validate_archive_catalogue(settings.parking_data_root, deep=deep)
    valid_count = sum(result.valid for result in results)
    return {
        "data_root": str(settings.parking_data_root),
        "valid": valid_count == len(results),
        "valid_count": valid_count,
        "expected_count": len(results),
        "archives": [result.as_dict() for result in results],
    }
