from __future__ import annotations

from fastapi import APIRouter
from sqlalchemy import text

from app.core.config import get_settings
from app.db.session import SessionLocal
from app.ml.model_store import model_status
from app.services.catalogue_service import CatalogueRepository, ScenarioNotFoundError
from app.services.demo_service import readiness_payload, select_showcase_scenarios

router = APIRouter()


def _context() -> tuple[CatalogueRepository, dict[str, object], list[dict[str, object]]]:
    settings = get_settings()
    repository = CatalogueRepository(settings.parking_data_root)
    catalogue = repository.load()
    scenarios = catalogue.get("scenarios", [])
    selected = select_showcase_scenarios(scenarios if isinstance(scenarios, list) else [])
    return repository, catalogue, selected


@router.get("/readiness")
def readiness() -> dict[str, object]:
    settings = get_settings()
    repository, catalogue, selected = _context()
    try:
        images_available = bool(selected) and all(
            repository.image_path(str(row["id"])).is_file() for row in selected
        )
    except (KeyError, ScenarioNotFoundError):
        images_available = False
    try:
        with SessionLocal() as session:
            database_connected = session.execute(text("SELECT 1")).scalar_one() == 1
    except Exception:  # pragma: no cover - defensive readiness boundary
        database_connected = False
    return readiness_payload(
        catalogue_prepared=bool(catalogue.get("prepared")),
        scenario_count=int(catalogue.get("scenario_count", 0)),
        selected=selected,
        model=model_status(settings.model_root),
        database_connected=database_connected,
        images_available=images_available,
    )


@router.get("/showcase")
def showcase() -> dict[str, object]:
    _, _, selected = _context()
    return {"count": len(selected), "scenarios": selected}
