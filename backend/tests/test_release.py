from copy import deepcopy

from fastapi.testclient import TestClient

from app.main import app
from app.release import (
    APPLICATION_VERSION,
    FITTING_POLICY,
    MODEL_NAME,
    release_manifest,
    validate_release_state,
)


def _valid_metadata() -> dict[str, object]:
    return {
        "model_name": MODEL_NAME,
        "training_samples": 6000,
        "validation_samples": 1800,
        "test_samples": 1800,
        "datasets_used_for_training": ["ACPDS", "CNRPark+EXT", "PKLot"],
        "fitting_policy": FITTING_POLICY,
        "independent_benchmark": {"unseen_test": {"unique_samples": 1800, "accuracy": 0.906667}},
    }


def test_release_manifest_is_the_final_offline_contract() -> None:
    manifest = release_manifest()

    assert manifest["version"] == "1.0.0"
    assert manifest["stage"] == "final"
    assert manifest["datasets"] == ["PKLot", "CNRPark+EXT", "ACPDS"]
    assert manifest["boundaries"] == {
        "hardware": False,
        "live_data": False,
        "cloud_ai": False,
    }


def test_release_endpoint_exposes_version_1_contract() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/system/release")

    assert response.status_code == 200
    assert response.json() == release_manifest()
    assert app.version == APPLICATION_VERSION


def test_release_state_accepts_verified_model_and_readiness() -> None:
    report = validate_release_state(
        version_text="1.0.0\n",
        backend_version="1.0.0",
        frontend_version="1.0.0",
        readiness={"ready": True, "summary": {"passed": 7, "total": 7}},
        model_name=MODEL_NAME,
        threshold=0.55,
        metadata=_valid_metadata(),
    )

    assert report["ready"] is True
    assert report["summary"] == {"passed": 7, "total": 7}


def test_release_state_rejects_benchmark_drift() -> None:
    metadata = deepcopy(_valid_metadata())
    benchmark = metadata["independent_benchmark"]
    assert isinstance(benchmark, dict)
    unseen_test = benchmark["unseen_test"]
    assert isinstance(unseen_test, dict)
    unseen_test["accuracy"] = 1.0

    report = validate_release_state(
        version_text="1.0.0",
        backend_version="1.0.0",
        frontend_version="1.0.0",
        readiness={"ready": True, "summary": {"passed": 7, "total": 7}},
        model_name=MODEL_NAME,
        threshold=0.55,
        metadata=metadata,
    )

    assert report["ready"] is False
    failed = [check["key"] for check in report["checks"] if not check["ready"]]
    assert failed == ["evaluation_contract"]


def test_release_state_rejects_package_version_drift() -> None:
    report = validate_release_state(
        version_text="1.0.0",
        backend_version="1.0.0",
        frontend_version="0.1.0",
        readiness={"ready": True, "summary": {"passed": 7, "total": 7}},
        model_name=MODEL_NAME,
        threshold=0.55,
        metadata=_valid_metadata(),
    )

    assert report["ready"] is False
    version_check = next(check for check in report["checks"] if check["key"] == "version")
    assert version_check["ready"] is False
    assert version_check["detail"]["frontend_package"] == "0.1.0"
