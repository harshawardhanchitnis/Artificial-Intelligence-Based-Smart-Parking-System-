"""The benchmark and product contracts, locked down at the API boundary.

Benchmark mode must keep asserting accuracy against verified labels, and product
mode must never assert accuracy about an image nobody has labelled.  These are
opposite failure modes and both were live before this work: the scenario path
had no mode label at all, and the history view counted every space on a user
upload as an incorrect prediction because the ground-truth field was absent.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _first_scenario(client: TestClient) -> str | None:
    response = client.get("/api/v1/datasets/scenarios", params={"limit": 1})
    if response.status_code != 200:
        return None
    scenarios = response.json().get("scenarios") or []
    return str(scenarios[0]["id"]) if scenarios else None


def test_benchmark_mode_is_labelled_and_claims_ground_truth(client: TestClient) -> None:
    scenario = _first_scenario(client)
    if scenario is None:
        pytest.skip("No prepared scenarios are available on this machine")
    response = client.post(f"/api/v1/analysis/scenarios/{scenario}")
    if response.status_code == 409:
        pytest.skip("Occupancy model is not installed on this machine")
    assert response.status_code == 200
    payload = response.json()

    assert payload["analysis_mode"] == "benchmark"
    assert payload["ground_truth_agreement"] is not None
    # Benchmark mode scores the bays the dataset defines and adds nothing.
    assert payload["total_spaces"] == len(payload["predictions"])
    assert "vehicles" not in payload
    for prediction in payload["predictions"]:
        assert "ground_truth_occupied" in prediction
        assert "correct" in prediction


def test_benchmark_counts_reconcile_with_the_defined_slots(client: TestClient) -> None:
    scenario = _first_scenario(client)
    if scenario is None:
        pytest.skip("No prepared scenarios are available on this machine")
    response = client.post(f"/api/v1/analysis/scenarios/{scenario}")
    if response.status_code == 409:
        pytest.skip("Occupancy model is not installed on this machine")
    payload = response.json()
    assert payload["occupied_spaces"] + payload["vacant_spaces"] == payload["total_spaces"]


def test_history_detail_of_an_upload_carries_no_correctness_claim(
    client: TestClient,
) -> None:
    """A record with no verified labels must not report per-space correctness.

    The interface derives "incorrect slots" from this field, so an upload that
    carried a ``correct`` key would have its accuracy reported against labels
    that do not exist.
    """
    # The whole window the endpoint allows. A window of 50 made this test skip
    # itself as soon as a testing session added fifty benchmark runs, which is
    # exactly when it was most worth running.
    response = client.get("/api/v1/analysis/history", params={"limit": 200})
    assert response.status_code == 200
    uploads = [
        row
        for row in response.json()["analyses"]
        if row["source_type"] == "image_upload"
    ]
    if not uploads:
        pytest.skip("No image uploads have been analysed on this machine")
    detail = client.get(f"/api/v1/analysis/history/{uploads[0]['id']}")
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["ground_truth_agreement"] is None
    for prediction in payload["predictions"]:
        assert "correct" not in prediction
        assert "ground_truth_occupied" not in prediction
