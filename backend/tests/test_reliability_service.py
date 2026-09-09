import json
from types import SimpleNamespace

from sqlalchemy import create_engine

from app.db.session import engine
from app.services import reliability_service
from app.services.reliability_service import (
    READINESS_CHECK_COUNT,
    collect_readiness,
    database_check,
)


def test_database_check_runs_sqlite_integrity_probe() -> None:
    memory_engine = create_engine("sqlite://")
    result = database_check(memory_engine)

    assert result["ready"] is True
    assert "quick_check ok" in str(result["detail"])


def test_application_sqlite_connections_use_reliability_pragmas() -> None:
    with engine.connect() as connection:
        foreign_keys = connection.exec_driver_sql("PRAGMA foreign_keys").scalar_one()
        busy_timeout = connection.exec_driver_sql("PRAGMA busy_timeout").scalar_one()
        journal_mode = connection.exec_driver_sql("PRAGMA journal_mode").scalar_one()

    assert foreign_keys == 1
    assert busy_timeout == 30_000
    assert str(journal_mode).lower() in {"wal", "memory"}


def test_collect_readiness_requires_every_local_dependency(tmp_path, monkeypatch) -> None:
    image = tmp_path / "demo" / "media" / "scenario.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"local-image")
    scenarios = [
        {
            "id": f"scenario-{index}",
            "dataset": dataset,
            "total_spaces": 10,
            "image_path": "demo/media/scenario.jpg",
        }
        for index, dataset in enumerate(("PKLot", "CNRPark+EXT", "ACPDS"), start=1)
    ]
    catalogue = {
        "prepared": True,
        "scenario_count": len(scenarios),
        "scenarios": scenarios,
    }
    (tmp_path / "demo" / "catalogue.json").write_text(json.dumps(catalogue), encoding="utf-8")
    monkeypatch.setattr(
        reliability_service,
        "model_status",
        lambda _root: {
            "ready": True,
            "model_name": "parking-occupancy-logistic-v2",
            "independent_benchmark": {"unseen_test": {"unique_samples": 1800}},
            "enhanced_occupancy": {"ready": True, "model_name": "enhanced"},
            "slot_localizer": {"ready": True, "model_name": "localizer"},
            "space_detector": {"ready": True, "model_name": "detector"},
        },
    )
    # Vehicle evidence and the fitted fusion are dependencies of the product
    # pipeline, so readiness reports them; the runtime still degrades to
    # bay-only occupancy when they are absent.
    monkeypatch.setattr(
        reliability_service,
        "vehicle_detector_status",
        lambda _root: {"ready": True, "model_name": "vehicle-detector-v3", "input_size": 1280},
    )
    monkeypatch.setattr(
        reliability_service,
        "load_policy",
        lambda _root: SimpleNamespace(fitted=True),
    )

    report = collect_readiness(tmp_path, tmp_path / "models", create_engine("sqlite://"))

    assert report["ready"] is True
    assert report["summary"] == {
        "passed": READINESS_CHECK_COUNT,
        "total": READINESS_CHECK_COUNT,
    }
    assert report["boundaries"] == {"hardware": False, "live_data": False, "cloud_ai": False}


def test_corrupt_catalogue_becomes_readiness_failure(tmp_path, monkeypatch) -> None:
    (tmp_path / "demo").mkdir(parents=True)
    (tmp_path / "demo" / "catalogue.json").write_text("not-json", encoding="utf-8")
    monkeypatch.setattr(reliability_service, "model_status", lambda _root: {"ready": False})

    report = collect_readiness(tmp_path, tmp_path / "models", create_engine("sqlite://"))

    assert report["ready"] is False
    catalogue = next(check for check in report["checks"] if check["key"] == "catalogue")
    assert catalogue["ready"] is False
    assert catalogue["detail"] == "JSONDecodeError"
