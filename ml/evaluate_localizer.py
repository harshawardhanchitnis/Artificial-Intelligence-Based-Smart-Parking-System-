"""Evaluate a trained parking-space detector on known and unseen cameras.

The headline number for this work is not overall accuracy but the gap between
cameras the detector was fitted on and cameras it has never seen, so both are
always reported together.  ``test_unseen`` is reporting-only and is read solely
when ``--include-test`` is passed.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

DEFAULT_DATA = Path("D:/Projects/AI Based Smart Parking System Data/prepared/obb-single")


def evaluate(model, data_yaml: Path, split_name: str, imgsz: int) -> dict[str, object]:
    started = time.perf_counter()
    metrics = model.val(data=str(data_yaml), imgsz=imgsz, split="val", verbose=False)
    elapsed = time.perf_counter() - started
    box = metrics.box
    precision, recall = float(box.mp), float(box.mr)
    return {
        "split": split_name,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / (precision + recall), 6)
        if precision + recall
        else 0.0,
        "map50": round(float(box.map50), 6),
        "map50_95": round(float(box.map), 6),
        "seconds": round(elapsed, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(arguments.weights))
    splits = [
        ("val_known", arguments.data / "data-known.yaml"),
        ("val_unseen", arguments.data / "data.yaml"),
    ]
    if arguments.include_test:
        splits.append(("test_unseen", arguments.data / "data-test.yaml"))

    results = [evaluate(model, path, name, arguments.imgsz) for name, path in splits]
    known = next((row for row in results if row["split"] == "val_known"), None)
    unseen = next((row for row in results if row["split"] == "val_unseen"), None)
    report = {
        "weights": str(arguments.weights),
        "imgsz": arguments.imgsz,
        "splits": results,
        "generalisation_gap_map50": (
            round(float(known["map50"]) - float(unseen["map50"]), 6)
            if known and unseen
            else None
        ),
    }
    destination = arguments.out or arguments.weights.parent.parent / "evaluation.json"
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
