"""Presentation Mode must not become a second, divergent source of numbers.

A manual comparison read Presentation Mode's "PKLot · PUCPR — 100 vacant, 0
occupied" against Analyse's "PUCPR · Sunny · 06-17-00 — 98 vacant, 2 occupied"
and concluded the showcase was serving fake data. It was not: those are two
different photographs of the same car park, and the showcase had simply never
disclosed *which* one it ran.

The confusion was possible because nothing pinned the relationship, so these
tests pin it:

* the showcase serves scenario *identifiers*, never precomputed results;
* running a showcase scenario through the ordinary benchmark endpoint is the
  only way its numbers are produced, so the two paths cannot drift;
* an agreement percentage must be reconcilable with the counts printed beside
  it, which is what makes "100% agreement" auditable rather than asserted.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as value:
        yield value


@pytest.fixture(scope="module")
def showcase(client):
    response = client.get("/api/v1/demo/showcase")
    assert response.status_code == 200
    scenarios = response.json()["scenarios"]
    if not scenarios:
        pytest.skip("no prepared showcase scenarios in this environment")
    return scenarios


def analyse(client, scenario_id: str) -> dict:
    """The one production benchmark path. Presentation Mode calls exactly this."""
    response = client.post(f"/api/v1/analysis/scenarios/{scenario_id}")
    assert response.status_code == 200, response.text
    return response.json()


def test_showcase_serves_identifiers_not_results(showcase) -> None:
    """The showcase may describe a scene and its labels -- never a prediction."""
    forbidden = {
        "predicted_occupied",
        "predictions",
        "ground_truth_agreement",
        "average_confidence",
        "analysis_id",
        "processing_time_ms",
        "model_name",
    }
    for scenario in showcase:
        assert scenario["id"], "a showcase entry must name the scenario it runs"
        leaked = forbidden & set(scenario)
        assert not leaked, (
            f"{scenario['id']} ships precomputed result fields {sorted(leaked)}; "
            "Presentation Mode must obtain those by running the analysis"
        )


def test_every_showcase_scenario_matches_the_benchmark_path(client, showcase) -> None:
    """Presentation ↔ Analyse parity, field by field, for every showcase dataset.

    Two runs of the same scenario differ only in their saved id and timing, so
    everything else must be identical -- including on a second run, which is
    what catches a cached or fixture-backed result.
    """
    for scenario in showcase:
        first = analyse(client, scenario["id"])
        second = analyse(client, scenario["id"])

        assert first["scenario_id"] == scenario["id"]
        assert first["dataset"] == scenario["dataset"]
        assert first["total_spaces"] == scenario["total_spaces"], (
            f"{scenario['id']}: the showcase advertises {scenario['total_spaces']} "
            f"spaces but the analysis scored {first['total_spaces']}"
        )
        for field in (
            "total_spaces",
            "vacant_spaces",
            "occupied_spaces",
            "ground_truth_agreement",
            "average_confidence",
            "model_name",
            "decision_threshold",
            "ground_truth_occupied_spaces",
            "ground_truth_vacant_spaces",
        ):
            assert first[field] == second[field], (
                f"{scenario['id']}: {field} changed between two runs of the same "
                f"scenario ({first[field]!r} then {second[field]!r})"
            )
        assert first["analysis_id"] != second["analysis_id"], (
            "each run must be saved as its own analysis record"
        )


def test_occupancy_counts_reconcile_with_the_total(client, showcase) -> None:
    for scenario in showcase:
        result = analyse(client, scenario["id"])
        uncertain = result.get("uncertain_spaces", 0) or 0
        assert (
            result["vacant_spaces"] + result["occupied_spaces"] + uncertain
            == result["total_spaces"]
        ), scenario["id"]


def test_agreement_cannot_contradict_the_counts_beside_it(client, showcase) -> None:
    """The specific claim a reader has to be able to trust.

    Each disagreeing bay moves the predicted occupied count by at most one, so
    the gap between prediction and labels can never exceed the disagreement
    count -- and at 100% agreement the two must be equal. Without this, a panel
    could show 100 vacant / 0 occupied beside 100% agreement while the labels
    said 98 / 2.
    """
    for scenario in showcase:
        result = analyse(client, scenario["id"])
        total = result["total_spaces"]
        disagreeing = result["disagreeing_spaces"]
        assert disagreeing == round((1 - result["ground_truth_agreement"]) * total)
        assert abs(result["occupied_spaces"] - result["ground_truth_occupied_spaces"]) <= (
            disagreeing
        ), scenario["id"]
        if result["ground_truth_agreement"] == 1.0:
            assert result["occupied_spaces"] == result["ground_truth_occupied_spaces"]
            assert result["vacant_spaces"] == result["ground_truth_vacant_spaces"]


def test_showcase_labels_match_the_catalogue_they_came_from(client, showcase) -> None:
    """The truth counts on the showcase card must be the dataset's own."""
    for scenario in showcase:
        response = client.get(f"/api/v1/datasets/scenarios/{scenario['id']}")
        assert response.status_code == 200
        catalogue = response.json()
        for field in ("total_spaces", "vacant_spaces", "occupied_spaces", "lot", "condition"):
            assert scenario[field] == catalogue[field], (
                f"{scenario['id']}: showcase {field} disagrees with the catalogue"
            )


def test_two_scenarios_from_one_lot_stay_distinguishable(client) -> None:
    """The trap that produced the report: same lot, different day, different truth.

    PUCPR cloudy is an empty lot; PUCPR sunny has two cars in it. They are not
    interchangeable, and any UI that shows one while naming the other is
    misleading even when both results are individually correct.
    """
    cloudy = "pklot-pucpr-cloudy-2012-09-12-06-31-24"
    sunny = "pklot-pucpr-sunny-2012-09-15-06-17-00"
    available = {row["id"] for row in client.get("/api/v1/datasets/scenarios").json()["scenarios"]}
    if not {cloudy, sunny} <= available:
        pytest.skip("the prepared catalogue does not hold both PUCPR scenarios")

    first, second = analyse(client, cloudy), analyse(client, sunny)
    assert first["ground_truth_occupied_spaces"] == 0
    assert second["ground_truth_occupied_spaces"] == 2
    assert first["occupied_spaces"] != second["occupied_spaces"], (
        "two different scenes produced identical occupancy; a fixture is being served"
    )


# --- showcase curation -----------------------------------------------------
#
# The showcase used to pick "the most spaces", which chose an empty PUCPR lot:
# 100 spaces, 100 vacant, nothing occupied. A correct result that demonstrated
# only half the product. Curation is now explicit, and these tests keep it
# honest -- it must select on the catalogue's own labels, never on how well the
# model happens to score.

def test_every_showcase_scene_demonstrates_both_states() -> None:
    from app.core.config import get_settings
    from app.services.catalogue_service import CatalogueRepository
    from app.services.demo_service import select_showcase_scenarios

    rows = CatalogueRepository(get_settings().parking_data_root).list_scenarios()
    if not rows:
        pytest.skip("no prepared catalogue in this environment")
    for scenario in select_showcase_scenarios(rows):
        assert scenario["occupied_spaces"] > 0, (
            f"{scenario['id']} has no occupied space, so the showcase cannot "
            "demonstrate occupied-space detection with it"
        )
        assert scenario["vacant_spaces"] > 0, (
            f"{scenario['id']} has no vacant space, so the showcase cannot "
            "demonstrate free-space reporting with it"
        )


def test_showcase_never_features_a_scene_the_model_trained_on() -> None:
    from app.core.config import get_settings
    from app.services.catalogue_service import CatalogueRepository
    from app.services.demo_service import select_showcase_scenarios

    rows = CatalogueRepository(get_settings().parking_data_root).list_scenarios()
    if not rows:
        pytest.skip("no prepared catalogue in this environment")
    for scenario in select_showcase_scenarios(rows):
        assert str(scenario.get("split", "")).lower() != "train", (
            f"{scenario['id']} is training data; reporting agreement against it "
            "flatters the model for no reason"
        )


def test_curation_is_deterministic_and_ignores_catalogue_order() -> None:
    from app.services.demo_service import select_showcase_scenarios

    catalogue = [
        {"dataset": "PKLot", "id": "b", "total_spaces": 40, "vacant_spaces": 25,
         "occupied_spaces": 15},
        {"dataset": "PKLot", "id": "a", "total_spaces": 100, "vacant_spaces": 100,
         "occupied_spaces": 0},
        {"dataset": "PKLot", "id": "c", "total_spaces": 40, "vacant_spaces": 25,
         "occupied_spaces": 15},
    ]
    chosen = select_showcase_scenarios(catalogue)[0]
    assert chosen["id"] == "b", "a tie must break on the identifier, not on input order"
    assert select_showcase_scenarios(list(reversed(catalogue)))[0]["id"] == "b"


def test_curation_reads_labels_not_predictions() -> None:
    """A scene must not become the showcase pick by being easy for the model.

    The ranking function is handed only catalogue fields here. If it ever starts
    consulting a prediction, agreement or confidence, this call raises instead
    of quietly ranking on it.
    """
    from app.services.demo_service import _showcase_rank

    row = {"dataset": "PKLot", "id": "x", "total_spaces": 40, "vacant_spaces": 25,
           "occupied_spaces": 15, "split": "valid"}
    assert _showcase_rank(row) == (0, -15, -40, "x")
    # Adding flattering model output must not move it up or down the ranking.
    flattered = {**row, "ground_truth_agreement": 1.0, "average_confidence": 1.0}
    assert _showcase_rank(flattered) == _showcase_rank(row)
