from __future__ import annotations

from collections.abc import Iterable
from typing import Any

REQUIRED_DATASETS = ("PKLot", "CNRPark+EXT", "ACPDS")


def _minority_class_count(row: dict[str, object]) -> int:
    """How many spaces the *rarer* of the two verified states holds.

    This is the single number that says whether a scene can demonstrate the
    product. A car park labelled 100 vacant and 0 occupied scores 0: it is a
    perfectly valid benchmark and shows nothing about occupied-space detection,
    because there is no occupied space in it to detect. A 25/15 split scores 15,
    and both states are visible at a glance.
    """
    return min(int(row.get("vacant_spaces", 0)), int(row.get("occupied_spaces", 0)))


def _showcase_rank(row: dict[str, object]) -> tuple[int, int, int, str]:
    """Ordering key for showcase curation. Lower sorts better.

    Deliberately blind to model output. Every term reads the catalogue's own
    verified labels and metadata, so the choice cannot drift toward whichever
    scene the model happens to score well on -- which would turn a demo into a
    quiet accuracy advertisement.

    In order:

    1. Never showcase a scene the model was trained on, while any alternative
       exists. Reporting agreement against training data flatters the system
       for no reason.
    2. Prefer the largest minority-class count, so both vacant and occupied are
       present in meaningful numbers rather than one token car.
    3. Prefer more total spaces, because a bigger lot shows more of the geometry
       the product is actually reasoning about.
    4. Fall back to the identifier, so the selection is fully deterministic and
       the same three scenes appear on every machine and every run.
    """
    is_training = 1 if str(row.get("split", "")).lower() == "train" else 0
    return (
        is_training,
        -_minority_class_count(row),
        -int(row.get("total_spaces", 0)),
        str(row.get("id", "")),
    )


def select_showcase_scenarios(
    scenarios: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Select one stable, information-rich scenario for each approved dataset.

    "Information-rich" used to mean "the most spaces", which picked an empty
    PUCPR lot for PKLot: 100 spaces, 100 vacant, nothing occupied. It was a
    correct result that demonstrated only half the product, and it read as a bug
    next to any other run of the same car park.
    """
    rows = [row for row in scenarios if isinstance(row, dict)]
    selected: list[dict[str, object]] = []
    for dataset in REQUIRED_DATASETS:
        candidates = [row for row in rows if row.get("dataset") == dataset]
        if candidates:
            selected.append(min(candidates, key=_showcase_rank))
    return selected


def readiness_payload(
    *,
    catalogue_prepared: bool,
    scenario_count: int,
    selected: list[dict[str, object]],
    model: dict[str, object],
    database_connected: bool,
    images_available: bool,
) -> dict[str, Any]:
    selected_datasets = {str(row.get("dataset")) for row in selected}
    coverage_ready = selected_datasets == set(REQUIRED_DATASETS)
    checks = [
        {
            "key": "catalogue",
            "label": "Prepared catalogue",
            "ready": catalogue_prepared and scenario_count > 0,
            "detail": f"{scenario_count} scenarios available",
        },
        {
            "key": "coverage",
            "label": "Dataset coverage",
            "ready": coverage_ready,
            "detail": f"{len(selected_datasets)}/{len(REQUIRED_DATASETS)} showcase datasets",
        },
        {
            "key": "images",
            "label": "Scenario images",
            "ready": images_available,
            "detail": "Featured images available locally" if images_available else "Image missing",
        },
        {
            "key": "model",
            "label": "Local AI model",
            "ready": bool(model.get("ready")),
            "detail": str(model.get("model_name", "Model unavailable")),
        },
        {
            "key": "database",
            "label": "Analysis database",
            "ready": database_connected,
            "detail": "SQLite connected" if database_connected else "SQLite unavailable",
        },
    ]
    return {
        "ready": all(bool(check["ready"]) for check in checks),
        "mode": "offline",
        "required_datasets": list(REQUIRED_DATASETS),
        "checks": checks,
        "showcase_count": len(selected),
        "boundaries": {
            "hardware": False,
            "live_data": False,
            "cloud_ai": False,
        },
    }
