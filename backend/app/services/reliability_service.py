from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import Engine, text

from app.ml.model_store import model_status
from app.ml.occupancy_fusion import load_policy
from app.ml.vehicle_detector import detector_status as vehicle_detector_status
from app.services.catalogue_service import CatalogueRepository, ScenarioNotFoundError
from app.services.demo_service import REQUIRED_DATASETS, select_showcase_scenarios
from app.services.video_service import resolve_ffmpeg

# Keeps readiness expectations in tests aligned with the checks actually run.
READINESS_CHECK_COUNT = 13


def describe_location(path: Path) -> str:
    """Name a configured location without publishing where it is on disk.

    Readiness is served unauthenticated, and an absolute path discloses the
    account name and directory layout of the machine to anyone who can reach
    the endpoint.  The operator already knows where they pointed the setting;
    what readiness has to answer is whether it resolved, so it reports the leaf
    name and leaves the rest out.
    """
    return f"...{Path(path).name}" if Path(path).name else "configured location"


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
            describe_location(data_root) if root_ready else "Folder not found",
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
    enhanced = model.get("enhanced_occupancy", {})
    localizer = model.get("slot_localizer", {})
    checks.append(
        _check(
            "automatic_image_ai",
            "Automatic image analysis models",
            bool(enhanced.get("ready")) and bool(localizer.get("ready")),
            (
                f"{enhanced.get('model_name')} + {localizer.get('model_name')}"
                if enhanced.get("ready") and localizer.get("ready")
                else "Enhanced occupancy and slot-localisation models are not both installed"
            ),
        )
    )
    media_root = data_root / "media"
    try:
        media_root.mkdir(parents=True, exist_ok=True)
        probe = media_root / ".readiness-probe"
        probe.write_bytes(b"ready")
        probe.unlink()
        media_ready = True
    except OSError:
        media_ready = False
    checks.append(
        _check(
            "media_storage",
            "Media processing storage",
            media_ready,
            describe_location(media_root) if media_ready else "Storage is not writable",
        )
    )
    detector = model.get("space_detector", {})
    checks.append(
        _check(
            "space_detector",
            "Generalized parking-space detector",
            bool(detector.get("ready")),
            str(detector.get("model_name"))
            if detector.get("ready")
            else str(detector.get("reason", "Not installed")),
        )
    )
    vehicles = vehicle_detector_status(model_root)
    checks.append(
        _check(
            "vehicle_detector",
            "Full-scene vehicle detector",
            bool(vehicles.get("ready")),
            # Vehicle evidence is optional by design: without it the system
            # falls back to bay-only occupancy rather than failing, so the
            # detail says what is lost rather than reporting a fault.
            f"{vehicles.get('model_name')} at {vehicles.get('input_size')}px"
            if vehicles.get("ready")
            else str(vehicles.get("reason", "Not installed")),
        )
    )
    fusion = load_policy(model_root)
    checks.append(
        _check(
            "occupancy_fusion",
            "Occupancy evidence fusion",
            fusion.fitted,
            "Fitted coefficients loaded"
            if fusion.fitted
            else "Not fitted; occupancy falls back to the classifier alone",
        )
    )
    ffmpeg_binary = resolve_ffmpeg()
    checks.append(
        _check(
            "video_encoder",
            "Video encoder (FFmpeg)",
            ffmpeg_binary is not None,
            "Available"
            if ffmpeg_binary is not None
            else "FFmpeg was not found; set FFMPEG_PATH or install it to enable video playback",
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

    assert len(checks) == READINESS_CHECK_COUNT
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
