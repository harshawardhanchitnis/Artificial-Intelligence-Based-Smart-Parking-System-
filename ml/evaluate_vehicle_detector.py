"""Score full-scene vehicle detectors against parking ground truth.

None of the three prepared datasets carries vehicle bounding boxes, so there is
no direct per-class ground truth to score against and this script does not
pretend otherwise.  What the datasets do carry is a verified occupancy label for
every annotated bay, and that supports two measurements that matter more to this
product than a COCO mAP would:

``occupied_bay_recall``   share of bays labelled occupied where the detector
                          finds a supported vehicle.  A vehicle standing in a
                          bay the system knows about should never be missed.
``vacant_bay_false_rate`` share of bays labelled vacant where the detector
                          nonetheless places a vehicle.  This is the error that
                          would turn a free space into a phantom occupied one.

Both are computed through ``app.ml.association``, the same asymmetric coverage
rules the runtime uses, so the benchmark measures the pipeline rather than a
detector in isolation.

Per-class precision and recall for CAR / TWO_WHEELER / TRUCK cannot be derived
from these labels at all.  The class distribution is reported so the mapping can
be inspected, and per-class accuracy is established separately on a small
hand-adjudicated subset -- see ``ml/review_vehicle_classes.py``.
"""

from __future__ import annotations

import argparse
import collections
import json
import statistics
import time
from pathlib import Path

from PIL import Image

DEFAULT_MANIFEST = Path(
    "D:/Projects/AI Based Smart Parking System Data/prepared/v2-protocol/source-manifest.jsonl"
)


def camera_family(stem: str) -> str:
    if "GOPR" in stem:
        return "ACPDS"
    if stem.startswith("camera"):
        return "CNRPark+EXT"
    return "PKLot"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="0")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.ml.association import associate
    from app.ml.vehicle_taxonomy import PRODUCT_CLASSES, product_class
    from ultralytics import YOLO

    model = YOLO(arguments.weights)
    model.to(arguments.device)
    names = model.names

    rows = [
        json.loads(line) for line in arguments.manifest.read_text(encoding="utf-8").splitlines()
    ]
    stems = {str(row["source_id"]).replace("/", "__").replace(" ", "_"): row for row in rows}

    tally: collections.Counter = collections.Counter()
    class_counts: collections.Counter = collections.Counter()
    latencies: list[float] = []
    images = 0

    paths = sorted(arguments.images.glob("*.jpg"))
    if arguments.limit:
        # Sample evenly across camera families.  The directory sorts by name,
        # which groups the families together, so a plain head would score one
        # dataset and call it the split.
        grouped: dict[str, list[Path]] = collections.defaultdict(list)
        for path in paths:
            grouped[camera_family(path.stem)].append(path)
        share = max(1, arguments.limit // max(len(grouped), 1))
        sampled: list[Path] = []
        for family in sorted(grouped):
            members = grouped[family]
            step = max(1, len(members) // share)
            sampled.extend(members[::step][:share])
        paths = sorted(sampled)

    for image_path in paths:
        row = stems.get(image_path.stem)
        if row is None:
            continue
        with Image.open(image_path) as handle:
            width, height = handle.size
        family = camera_family(image_path.stem)

        started = time.perf_counter()
        result = model.predict(
            source=str(image_path),
            imgsz=arguments.imgsz,
            conf=arguments.conf,
            verbose=False,
            device=arguments.device,
        )[0]
        latencies.append((time.perf_counter() - started) * 1000)
        images += 1

        vehicles: list[tuple[float, float, float, float]] = []
        for box, class_index in zip(
            result.boxes.xyxy.cpu().numpy(), result.boxes.cls.cpu().numpy(), strict=True
        ):
            mapped = product_class(str(names[int(class_index)]))
            if mapped is None:
                continue
            vehicles.append(tuple(float(value) for value in box))  # type: ignore[arg-type]
            class_counts[(family, mapped)] += 1
            class_counts[("ALL", mapped)] += 1

        slots: list[list[list[float]]] = []
        states: list[str] = []
        for slot in row["slots"]:
            occupied = slot.get("occupied")
            if occupied is None:
                continue
            slots.append([[x * width, y * height] for x, y in slot["polygon"]])
            states.append("occupied" if occupied else "vacant")

        association = associate(slots, vehicles)
        for slot_index, state in enumerate(states):
            found = bool(association.occupying_links(slot_index))
            for key in (family, "ALL"):
                tally[(key, state, "bays")] += 1
                if found:
                    tally[(key, state, "vehicle_found")] += 1

        # Precision, restricted to detections a labelled bay can adjudicate.
        # A detection occupying a bay the dataset calls vacant is a genuine
        # false positive; one occupying an occupied bay is a genuine true
        # positive.  Detections outside every bay are left out rather than
        # guessed at -- many are real vehicles the dataset does not label.
        owner: dict[int, str] = {}
        for slot_index, state in enumerate(states):
            for link in association.occupying_links(slot_index):
                # An occupied bay wins: a vehicle straddling both is real.
                if owner.get(link.vehicle_index) != "occupied":
                    owner[link.vehicle_index] = state
        for verdict in owner.values():
            for key in (family, "ALL"):
                tally[(key, "adjudicated", "true" if verdict == "occupied" else "false")] += 1

        for key in (family, "ALL"):
            tally[(key, "any", "vehicles")] += len(vehicles)
            tally[(key, "any", "unmapped")] += len(association.unmapped_vehicles)
            tally[(key, "any", "images")] += 1

    groups = []
    for key in sorted({name for name, _, _ in tally}):
        occupied = tally[(key, "occupied", "bays")]
        vacant = tally[(key, "vacant", "bays")]
        entry: dict[str, object] = {
            "group": key,
            "occupied_bays": occupied,
            "vacant_bays": vacant,
            "vehicles_detected": tally[(key, "any", "vehicles")],
            "unmapped_vehicles": tally[(key, "any", "unmapped")],
        }
        if occupied:
            entry["occupied_bay_recall"] = round(
                tally[(key, "occupied", "vehicle_found")] / occupied, 4
            )
        if vacant:
            entry["vacant_bay_false_rate"] = round(
                tally[(key, "vacant", "vehicle_found")] / vacant, 4
            )
        true_positives = tally[(key, "adjudicated", "true")]
        false_positives = tally[(key, "adjudicated", "false")]
        if true_positives + false_positives:
            entry["adjudicated_detections"] = true_positives + false_positives
            entry["precision_in_labelled_bays"] = round(
                true_positives / (true_positives + false_positives), 4
            )
        images_seen = tally[(key, "any", "images")]
        if images_seen:
            entry["false_detections_per_image"] = round(false_positives / images_seen, 4)
        entry["class_mix"] = {
            product: class_counts[(key, product)] for product in PRODUCT_CLASSES
        }
        groups.append(entry)

    report = {
        "candidate": arguments.label or Path(arguments.weights).stem,
        "weights": str(arguments.weights),
        "split": arguments.images.name,
        "imgsz": arguments.imgsz,
        "conf": arguments.conf,
        "device": arguments.device,
        "images": images,
        "latency_ms_mean": round(statistics.fmean(latencies), 2) if latencies else None,
        "latency_ms_median": round(statistics.median(latencies), 2) if latencies else None,
        "groups": groups,
    }
    print(json.dumps(report, indent=2))
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
