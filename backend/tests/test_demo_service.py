from app.services.demo_service import readiness_payload, select_showcase_scenarios


def test_showcase_selects_largest_scenario_per_dataset_in_fixed_order() -> None:
    scenarios = [
        {"id": "pklot-small", "dataset": "PKLot", "total_spaces": 10},
        {"id": "acpds", "dataset": "ACPDS", "total_spaces": 20},
        {"id": "cnr", "dataset": "CNRPark+EXT", "total_spaces": 30},
        {"id": "pklot-large", "dataset": "PKLot", "total_spaces": 40},
    ]
    selected = select_showcase_scenarios(reversed(scenarios))

    assert [row["dataset"] for row in selected] == ["PKLot", "CNRPark+EXT", "ACPDS"]
    assert selected[0]["id"] == "pklot-large"


def test_readiness_requires_every_local_component() -> None:
    selected = [
        {"dataset": "PKLot"},
        {"dataset": "CNRPark+EXT"},
        {"dataset": "ACPDS"},
    ]
    payload = readiness_payload(
        catalogue_prepared=True,
        scenario_count=8,
        selected=selected,
        model={"ready": True, "model_name": "local-model"},
        database_connected=True,
        images_available=True,
    )

    assert payload["ready"] is True
    assert payload["showcase_count"] == 3
    assert payload["boundaries"] == {"hardware": False, "live_data": False, "cloud_ai": False}


def test_readiness_reports_missing_dataset_coverage() -> None:
    payload = readiness_payload(
        catalogue_prepared=True,
        scenario_count=2,
        selected=[{"dataset": "PKLot"}],
        model={"ready": True},
        database_connected=True,
        images_available=True,
    )

    assert payload["ready"] is False
    coverage = next(check for check in payload["checks"] if check["key"] == "coverage")
    assert coverage["ready"] is False
