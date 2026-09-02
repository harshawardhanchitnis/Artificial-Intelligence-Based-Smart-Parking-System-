from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from app.core.config import get_settings
from app.datasets.archive_validator import validate_archive_catalogue
from app.services.catalogue_service import CatalogueRepository, ScenarioNotFoundError

router = APIRouter()


def repository() -> CatalogueRepository:
    return CatalogueRepository(get_settings().parking_data_root)


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


@router.get("/catalogue")
def catalogue() -> dict[str, object]:
    return repository().load()


@router.get("/scenarios")
def scenarios(
    dataset: str | None = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
) -> dict[str, object]:
    rows = repository().list_scenarios(dataset=dataset, limit=limit)
    return {"count": len(rows), "scenarios": rows}


@router.get("/scenarios/{scenario_id}")
def scenario(scenario_id: str) -> dict[str, object]:
    try:
        return repository().get_scenario(scenario_id)
    except ScenarioNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Scenario not found") from exc


@router.get("/scenarios/{scenario_id}/image", response_class=FileResponse)
def scenario_image(scenario_id: str) -> FileResponse:
    try:
        image_path = repository().image_path(scenario_id)
    except ScenarioNotFoundError as exc:
        raise HTTPException(status_code=404, detail="Scenario image not found") from exc
    return FileResponse(image_path, media_type="image/jpeg")
