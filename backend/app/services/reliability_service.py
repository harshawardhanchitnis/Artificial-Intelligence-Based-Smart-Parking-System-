from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine, text

from app.ml.model_store import model_status
from app.services.catalogue_service import CatalogueRepository, ScenarioNotFoundError
from app.services.demo_service import REQUIRED_DATASETS, select_showcase_scenarios


def _check(key: str, label: str, ready: bool, detail: str) -> dict[str, object]:
    return {"key": key, "label": label, "ready": ready, "detail": detail}


def database_check(engine: Engine) -> dict[str, object]:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            if engine.dialect.name == "sqlite":
                quick_check = str(connection.exec_driver_sql("PRAGMA quick_check").scalar_one())
                if quick_check.lower() != "ok":
                    return _check("database", "Analysis database", False, quick_check)
                user_version = int(connection.exec_driver_sql("PRAGMA user_version").scalar_one())
                return _check(
                    "database",
                    "Analysis database",
                    True,
                    f"SQLite quick_check ok; schema version {user_version}",
                )
        return _check("database", "Analysis database", True, "Database connection verified")
    except Exception as exc:  # pragma: no cover - exercised through readiness failure paths
        return _check("database", "Analysis database", False, type(exc).__name__)


def collect_readiness(data_root: Path, model_root: Path, engine: Engine) -> dict[str, object]:
    checks: list[dict[str, object]] = []
    root_ready = data_root.is_dir()
    checks.append(
        _check(
            "dataset_root",
            "External dataset root",
            root_ready,
            str(data_root) if root_ready else "Folder not found",
        )
    )
    checks.append(database_check(engine))

    repository = CatalogueRepository(data_root)
    catalogue_error = ""
    try:
        catalogue = repository.load()
    except (OSError, TypeError, ValueError) as exc:
        catalogue = {}
        catalogue_error = type(exc).__name__
    scenarios_value = catalogue.get("scenarios", [])
    scenarios = scenarios_value if isinstance(scenarios_value, list) else []
    scenario_count = len(scenarios)
    catalogue_ready = bool(catalogue.get("prepared")) and scenario_count > 0
    checks.append(
        _check(
            "catalogue",
            "Prepared demo catalogue",
            catalogue_ready,
            f"{scenario_count} scenarios"
            if catalogue_ready
            else catalogue_error or "Demo profile not prepared",
        )
    )

    selected = select_showcase_scenarios(scenarios)
    selected_datasets = {str(row.get("dataset")) for row in selected}
    coverage_ready = selected_datasets == set(REQUIRED_DATASETS)
    checks.append(
        _check(
            "dataset_coverage",
            "Showcase dataset coverage",
            coverage_ready,
            f"{len(selected_datasets)}/{len(REQUIRED_DATASETS)} approved datasets",
        )
    )

    try:
        images_ready = bool(selected) and all(
            repository.image_path(str(row["id"])).is_file() for row in selected
        )
    except (KeyError, OSError, ScenarioNotFoundError):
        images_ready = False
    checks.append(
        _check(
            "scenario_images",
            "Showcase scenario images",
            images_ready,
            "All featured images readable" if images_ready else "Featured image unavailable",
        )
    )

    model = model_status(model_root)
    model_ready = bool(model.get("ready"))
    checks.append(
        _check(
            "model",
            "Local AI model",
            model_ready,
            str(model.get("model_name") if model_ready else model.get("reason", "Not ready")),
        )
    )
    benchmark = model.get("independent_benchmark")
    unseen = benchmark.get("unseen_test") if isinstance(benchmark, dict) else None
    test_samples = unseen.get("unique_samples") if isinstance(unseen, dict) else None
    benchmark_ready = isinstance(test_samples, int) and test_samples > 0
    checks.append(
        _check(
            "independent_benchmark",
            "Independent unseen benchmark",
            benchmark_ready,
            f"{test_samples} unseen samples"
            if benchmark_ready
            else "Benchmark metadata unavailable",
        )
    )

    passed = sum(bool(check["ready"]) for check in checks)
    ready = passed == len(checks)
    return {
        "status": "ready" if ready else "not_ready",
        "ready": ready,
        "timestamp": datetime.now(UTC).isoformat(),
        "checks": checks,
        "summary": {"passed": passed, "total": len(checks)},
        "boundaries": {"hardware": False, "live_data": False, "cloud_ai": False},
    }
