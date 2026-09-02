from __future__ import annotations

import json
from pathlib import Path


class ScenarioNotFoundError(LookupError):
    """Raised when a scenario ID is not present in the prepared catalogue."""


class CatalogueRepository:
    def __init__(self, data_root: Path) -> None:
        self.data_root = data_root.resolve()
        self.catalogue_path = self.data_root / "demo" / "catalogue.json"

    def load(self) -> dict[str, object]:
        if not self.catalogue_path.is_file():
            return {
                "prepared": False,
                "profile": "demo",
                "scenario_count": 0,
                "datasets": [],
                "scenarios": [],
            }
        with self.catalogue_path.open(encoding="utf-8") as handle:
            catalogue = json.load(handle)
        catalogue["prepared"] = True
        return catalogue

    def list_scenarios(
        self, *, dataset: str | None = None, limit: int = 100
    ) -> list[dict[str, object]]:
        scenarios = self.load().get("scenarios", [])
        if not isinstance(scenarios, list):
            return []
        selected = [
            scenario
            for scenario in scenarios
            if isinstance(scenario, dict)
            and (dataset is None or scenario.get("dataset") == dataset)
        ]
        return selected[:limit]

    def get_scenario(self, scenario_id: str) -> dict[str, object]:
        for scenario in self.list_scenarios(limit=10_000):
            if scenario.get("id") == scenario_id:
                return scenario
        raise ScenarioNotFoundError(scenario_id)

    def image_path(self, scenario_id: str) -> Path:
        scenario = self.get_scenario(scenario_id)
        image_path = (self.data_root / str(scenario["image_path"])).resolve()
        if not image_path.is_relative_to(self.data_root) or not image_path.is_file():
            raise ScenarioNotFoundError(scenario_id)
        return image_path
