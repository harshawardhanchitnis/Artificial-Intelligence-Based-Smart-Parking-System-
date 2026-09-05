from __future__ import annotations

from pathlib import Path
from typing import Any

VERSION_FILE = Path(__file__).resolve().parents[2] / "VERSION"
APPLICATION_VERSION = VERSION_FILE.read_text(encoding="utf-8").strip()
RELEASE_LABEL = "Version 1.0"
RELEASE_STAGE = "final"
REQUIRED_DATASETS = ("PKLot", "CNRPark+EXT", "ACPDS")
MODEL_NAME = "parking-occupancy-logistic-v2"
DECISION_THRESHOLD = 0.55
TRAINING_SAMPLES = 6000
VALIDATION_SAMPLES = 1800
UNSEEN_TEST_SAMPLES = 1800
UNSEEN_TEST_ACCURACY = 0.906667
FITTING_POLICY = "train-only; validation selects threshold; test remains unseen"


def release_manifest() -> dict[str, Any]:
    """Return the immutable Version 1.0 product and evaluation contract."""
    return {
        "version": APPLICATION_VERSION,
        "label": RELEASE_LABEL,
        "stage": RELEASE_STAGE,
        "mode": "offline",
        "datasets": list(REQUIRED_DATASETS),
        "model": {
            "name": MODEL_NAME,
            "decision_threshold": DECISION_THRESHOLD,
            "training_samples": TRAINING_SAMPLES,
            "validation_samples": VALIDATION_SAMPLES,
            "unseen_test_samples": UNSEEN_TEST_SAMPLES,
            "unseen_test_accuracy": UNSEEN_TEST_ACCURACY,
            "fitting_policy": FITTING_POLICY,
        },
        "capabilities": [
            "prepared scenario catalogue",
            "local per-slot occupancy inference",
            "saved analysis history",
            "analytics and downloadable reports",
            "independent benchmark diagnostics",
            "guided three-dataset presentation",
        ],
        "boundaries": {
            "hardware": False,
            "live_data": False,
            "cloud_ai": False,
        },
    }


def validate_release_state(
    *,
    version_text: str,
    backend_version: str,
    frontend_version: str,
    readiness: dict[str, Any],
    model_name: str,
    threshold: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Compare an installed system with the immutable Version 1.0 contract."""
    benchmark = metadata.get("independent_benchmark", {})
    unseen_test = benchmark.get("unseen_test", {}) if isinstance(benchmark, dict) else {}
    observed_datasets = metadata.get("datasets_used_for_training", [])
    checks = [
        {
            "key": "version",
            "ready": version_text.strip()
            == backend_version
            == frontend_version
            == APPLICATION_VERSION,
            "detail": {
                "version_file": version_text.strip(),
                "backend_package": backend_version,
                "frontend_package": frontend_version,
                "application": APPLICATION_VERSION,
            },
        },
        {
            "key": "deep_readiness",
            "ready": bool(readiness.get("ready"))
            and readiness.get("summary") == {"passed": 7, "total": 7},
            "detail": readiness.get("summary"),
        },
        {
            "key": "model_identity",
            "ready": model_name == MODEL_NAME,
            "detail": model_name,
        },
        {
            "key": "decision_threshold",
            "ready": abs(float(threshold) - DECISION_THRESHOLD) < 1e-12,
            "detail": float(threshold),
        },
        {
            "key": "dataset_contract",
            "ready": sorted(str(item) for item in observed_datasets) == sorted(REQUIRED_DATASETS),
            "detail": observed_datasets,
        },
        {
            "key": "partition_contract",
            "ready": metadata.get("training_samples") == TRAINING_SAMPLES
            and metadata.get("validation_samples") == VALIDATION_SAMPLES
            and metadata.get("test_samples") == UNSEEN_TEST_SAMPLES,
            "detail": {
                "training": metadata.get("training_samples"),
                "validation": metadata.get("validation_samples"),
                "unseen_test": metadata.get("test_samples"),
            },
        },
        {
            "key": "evaluation_contract",
            "ready": metadata.get("fitting_policy") == FITTING_POLICY
            and unseen_test.get("unique_samples") == UNSEEN_TEST_SAMPLES
            and unseen_test.get("accuracy") == UNSEEN_TEST_ACCURACY,
            "detail": {
                "fitting_policy": metadata.get("fitting_policy"),
                "unique_samples": unseen_test.get("unique_samples"),
                "accuracy": unseen_test.get("accuracy"),
            },
        },
    ]
    passed = sum(bool(check["ready"]) for check in checks)
    return {
        "ready": passed == len(checks),
        "version": APPLICATION_VERSION,
        "checks": checks,
        "summary": {"passed": passed, "total": len(checks)},
    }
