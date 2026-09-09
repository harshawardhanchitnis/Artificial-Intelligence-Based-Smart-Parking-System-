"""Assemble the product acceptance scorecard from the measured artifacts.

Every row is read from a file some experiment wrote, so nothing here can drift
from what was actually measured.  A row whose artifact is missing prints
``not measured`` rather than a plausible-looking number: the point of the
scorecard is to make weak evidence visible, so an absent measurement has to look
absent.

Per-class vehicle rows are reported separately and never averaged, because the
corpus supports one of the three classes far better than the others and an
aggregate would hide exactly that.

    python ml/acceptance_table.py --artifacts <scratch>/artifacts
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

MISSING = "not measured"


def load(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def group(report: Any, name: str) -> dict[str, Any]:
    if not isinstance(report, dict):
        return {}
    for entry in report.get("groups", []):
        if entry.get("group") == name:
            return entry
    return {}


def show(value: Any, digits: int = 4, suffix: str = "") -> str:
    if value is None:
        return MISSING
    if isinstance(value, float):
        return f"{value:.{digits}f}{suffix}"
    return f"{value}{suffix}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--models", type=Path, default=Path(
        "D:/Projects/AI Based Smart Parking System Data/models"
    ))
    arguments = parser.parse_args()
    root = arguments.artifacts

    space = load(root / "tableA" / "selected-space-model.json") or load(
        root / "tableA" / "pose-geom-cpu.json"
    )
    vehicles = load(root / "tableB" / "final-operating-point.json")
    classes = load(root / "tableB" / "vehicle-classes-coco.json")
    fusion = load(arguments.models / "occupancy-fusion-v3.json")
    latency = load(root / "latency-profile.json")
    calibration = load(root / "calibration-summary.json")
    area = load(root / "parking-area-summary.json")

    space_all = group(space, "ALL")
    space_acpds = group(space, "ACPDS")
    vehicle_all = group(vehicles, "ALL")
    development = (fusion or {}).get("development", {})
    fused = next(
        (
            row
            for row in development.get("comparison", [])
            if "fused" in str(row.get("signal", ""))
        ),
        {},
    )

    def per_class(name: str, field: str) -> Any:
        for row in (classes or {}).get("classes", []):
            if row.get("class") == name:
                return row.get(field)
        return None

    rows: list[tuple[str, str, str]] = [
        (
            "A  Parking-space coverage",
            show(space_all.get("recall50")),
            "recall at polygon IoU 0.50, unseen cameras",
        ),
        (
            "B  Vacant-space coverage",
            show(group(space, "state:vacant").get("recall50")),
            "the figure a parking product lives on",
        ),
        (
            "C  Occupied-space coverage",
            show(group(space, "state:occupied").get("recall50")),
            "",
        ),
        (
            "D1 Vehicle recall - CAR",
            show(per_class("CAR", "recall")),
            "general-domain development set",
        ),
        (
            "D2 Vehicle recall - TWO_WHEELER",
            show(per_class("TWO_WHEELER", "recall")),
            "NOT parking-domain validated",
        ),
        (
            "D3 Vehicle recall - TRUCK",
            show(per_class("TRUCK", "recall")),
            "general-domain development set",
        ),
        ("E1 Vehicle precision - CAR", show(per_class("CAR", "precision")), ""),
        (
            "E2 Vehicle precision - TWO_WHEELER",
            show(per_class("TWO_WHEELER", "precision")),
            "NOT parking-domain validated",
        ),
        ("E3 Vehicle precision - TRUCK", show(per_class("TRUCK", "precision")), ""),
        (
            "F  Polygon geometry quality",
            show(space_acpds.get("mean_matched_polygon_iou")),
            "matched polygon IoU on geometry-faithful ACPDS",
        ),
        (
            "   Corner localisation error",
            show(space_acpds.get("mean_corner_error")),
            "fraction of bay scale, ACPDS",
        ),
        (
            "G  Unseen-camera generalisation",
            show(space_all.get("recall75")),
            "recall at the strict IoU 0.75",
        ),
        (
            "H  Automatic calibration",
            show((calibration or {}).get("halves_agreement"), suffix=""),
            "spaces recovered by disjoint halves at IoU 0.5",
        ),
        (
            "I  Parking-area understanding",
            show((area or {}).get("outside_excluded")),
            "off-site vehicles excluded from site figures",
        ),
        ("J  False-vacant rate", show(fused.get("false_vacant")), "fused, validation split"),
        ("K  False-occupied rate", show(fused.get("false_occupied")), "fused, validation split"),
        (
            "L  Uncertain rate",
            show(
                None
                if fused.get("decided_share") is None
                else 1 - float(fused["decided_share"])
            ),
            "share of bays where no verdict is asserted",
        ),
        (
            "M  False vehicle detections / image",
            show(vehicle_all.get("false_detections_per_image")),
            "adjudicated by labelled bays",
        ),
        (
            "N  CPU latency (total image)",
            show((latency or {}).get("cpu_total_ms"), 1, " ms"),
            "clean profile, no training running",
        ),
        (
            "O  GPU latency (total image)",
            show((latency or {}).get("gpu_total_ms"), 1, " ms"),
            "RTX 4070",
        ),
    ]

    width = max(len(name) for name, _, _ in rows)
    print("PRODUCT ACCEPTANCE SCORECARD")
    print("=" * (width + 34))
    for name, value, note in rows:
        print(f"{name:<{width}}  {value:>12}   {note}")
    print("=" * (width + 34))
    print(
        "Weak classes are reported on their own rows and never averaged into an "
        "aggregate."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
