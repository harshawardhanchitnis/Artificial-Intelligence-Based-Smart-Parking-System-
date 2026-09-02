from __future__ import annotations

from collections.abc import Iterable
from typing import Any

REQUIRED_DATASETS = ("PKLot", "CNRPark+EXT", "ACPDS")


def select_showcase_scenarios(
    scenarios: Iterable[dict[str, object]],
) -> list[dict[str, object]]:
    """Select one stable, information-rich scenario for each approved dataset."""
    rows = [row for row in scenarios if isinstance(row, dict)]
    selected: list[dict[str, object]] = []
    for dataset in REQUIRED_DATASETS:
        candidates = [row for row in rows if row.get("dataset") == dataset]
        if candidates:
            selected.append(
                min(
                    candidates,
                    key=lambda row: (
                        -int(row.get("total_spaces", 0)),
                        str(row.get("id", "")),
                    ),
                )
            )
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
