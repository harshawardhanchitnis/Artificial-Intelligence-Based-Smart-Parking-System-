"""Report benchmark-mode and product-mode results for the same scenes.

The two modes answer different questions and must be read separately, which is
easiest to see when they are printed side by side for one image.

Benchmark mode scores the bays the dataset defines against its verified labels.
Product mode establishes its own geometry and searches the whole frame for
vehicles, so it may legitimately find bays and vehicles the benchmark has no
labels for.  A product-mode count is never evidence about the benchmark: if the
dataset defines 76 bays and product mode proposes 95, the ground-truth agreement
figure still describes those 76 and nothing else.

    python ml/compare_modes.py --per-dataset 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-dataset", type=int, default=4)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.core.config import get_settings
    from app.db.models import Base
    from app.ml.inference import analyse_scenario
    from app.ml.parking_area import IN_PARKING_AREA, OUTSIDE_PARKING_AREA
    from app.ml.vehicle_detector import unclassified_count, vehicle_counts
    from app.services.catalogue_service import CatalogueRepository
    from app.services.scene_analysis import analyse_product
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    settings = get_settings()
    repository = CatalogueRepository(settings.parking_data_root)
    scenarios = repository.list_scenarios(limit=10_000)

    chosen: list[dict] = []
    for dataset in ("PKLot", "CNRPark+EXT", "ACPDS"):
        matching = [s for s in scenarios if str(s.get("dataset")) == dataset]
        step = max(1, len(matching) // max(arguments.per_dataset, 1))
        chosen.extend(matching[::step][: arguments.per_dataset])

    # An in-memory database: product mode needs a session for camera layouts,
    # and this must not touch the application's own history.
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    rows: list[dict[str, object]] = []
    for scenario in chosen:
        scenario_id = str(scenario["id"])
        benchmark = analyse_scenario(
            settings.parking_data_root, settings.model_root, scenario_id
        )
        with Image.open(repository.image_path(scenario_id)) as handle:
            image = handle.convert("RGB")
        with session_factory() as session:
            product = analyse_product(session, settings, image)

        counts = vehicle_counts(product.facility_vehicles)
        rows.append(
            {
                "scenario": scenario_id,
                "dataset": str(scenario["dataset"]),
                "benchmark": {
                    "defined_spaces": benchmark["total_spaces"],
                    "occupied": benchmark["occupied_spaces"],
                    "vacant": benchmark["vacant_spaces"],
                    "ground_truth_agreement": benchmark["ground_truth_agreement"],
                },
                "product": {
                    "layout_state": product.layout.state if product.layout else "supplied",
                    "outside_area": sum(
                        1 for place in product.placements if place == OUTSIDE_PARKING_AREA
                    ),
                    "unclassified": unclassified_count(product.facility_vehicles),
                    "detected_spaces": len(product.slots),
                    "occupied": product.occupied,
                    "vacant": product.vacant,
                    "uncertain": product.uncertain,
                    "cars": counts["CAR"],
                    "two_wheelers": counts["TWO_WHEELER"],
                    "trucks": counts["TRUCK"],
                    "unmapped_vehicles": sum(
                        1 for place in product.placements if place == IN_PARKING_AREA
                    ),
                },
            }
        )

    header = (
        f"{'scenario':<34}{'dataset':<13}| {'defined':>7}{'occ':>5}{'vac':>5}{'agree':>7} "
        f"| {'layout':<26}{'spaces':>7}{'occ':>5}{'vac':>5}{'unc':>5}"
        f"{'CAR':>5}{'2W':>4}{'TRK':>5}{'unmapd':>7}{'outside':>8}{'unclass':>8}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        b, p = row["benchmark"], row["product"]  # type: ignore[index]
        print(
            f"{str(row['scenario'])[:33]:<34}{row['dataset']!s:<13}| "
            f"{b['defined_spaces']:>7}{b['occupied']:>5}{b['vacant']:>5}"
            f"{b['ground_truth_agreement']:>7.3f} | {str(p['layout_state'])[:25]:<26}"
            f"{p['detected_spaces']:>7}{p['occupied']:>5}{p['vacant']:>5}{p['uncertain']:>5}"
            f"{p['cars']:>5}{p['two_wheelers']:>4}{p['trucks']:>5}"
            f"{p['unmapped_vehicles']:>7}{p['outside_area']:>8}{p['unclassified']:>8}"
        )
    print()
    print(
        "Benchmark columns describe the dataset's own bays only. Product columns "
        "describe the whole visible scene and are not evidence about the benchmark."
    )
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
