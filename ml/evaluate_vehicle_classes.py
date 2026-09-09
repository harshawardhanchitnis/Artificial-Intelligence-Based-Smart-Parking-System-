"""Per-class precision and recall for CAR / TWO_WHEELER / TRUCK.

The parking corpus cannot answer this question.  Across roughly a thousand
detections on unseen parking cameras it produced one two-wheeler, so any
per-class figure derived from it would be noise presented as a measurement.

This script uses a separate, general-domain development set instead -- COCO
val2017, which carries instance annotations for car, motorcycle and truck -- and
scores the detector through the product taxonomy: source labels are folded the
same way the runtime folds them, and annotations of classes the product does not
support are excluded from both sides so neither side is penalised for them.

Two limits are inherent and must travel with the numbers:

* COCO val2017 is the detector's own validation set.  These figures describe the
  detector's class ability, not its generalisation to unseen data.
* COCO is street photography.  A motorcycle there is metres from the lens; a
  motorcycle in a car park is thirty metres below a CCTV mast.  Good numbers
  here do **not** establish two-wheeler support in the parking domain.

The set is kept under a separate ``vehicle-dev`` root, entirely outside the
parking benchmark protocol, so nothing here can contaminate that evaluation.

    python ml/evaluate_vehicle_classes.py --weights weights/yolo11n.pt
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np

DEFAULT_ROOT = Path("D:/Projects/AI Based Smart Parking System Data/vehicle-dev")
MATCH_IOU = 0.5


def box_iou(first: np.ndarray, second: np.ndarray) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    overlap = max(right - left, 0.0) * max(bottom - top, 0.0)
    if overlap <= 0:
        return 0.0
    first_area = (first[2] - first[0]) * (first[3] - first[1])
    second_area = (second[2] - second[0]) * (second[3] - second[1])
    union = first_area + second_area - overlap
    return float(overlap / union) if union > 0 else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.ml.vehicle_taxonomy import PRODUCT_CLASSES, product_class
    from ultralytics import YOLO

    annotations_path = arguments.root / "annotations" / "instances_val2017.json"
    if not annotations_path.is_file():
        print(f"COCO annotations not found at {annotations_path}")
        return 1
    coco = json.loads(annotations_path.read_text(encoding="utf-8"))

    # Category id -> product class, dropping everything the product does not
    # expose.  A ``bus`` annotation is not ground truth for this system, so it
    # is removed from the truth side exactly as bus detections are removed from
    # the prediction side.
    category_to_product = {
        category["id"]: product_class(str(category["name"]))
        for category in coco["categories"]
        if product_class(str(category["name"])) is not None
    }
    truth: dict[int, list[tuple[str, np.ndarray]]] = collections.defaultdict(list)
    for annotation in coco["annotations"]:
        product = category_to_product.get(annotation["category_id"])
        if product is None or annotation.get("iscrowd"):
            continue
        x, y, width, height = annotation["bbox"]
        truth[annotation["image_id"]].append(
            (product, np.array([x, y, x + width, y + height], dtype=np.float64))
        )

    images = {image["id"]: image["file_name"] for image in coco["images"]}
    # Only images that contain at least one supported vehicle; the rest would
    # contribute nothing to recall and would dominate the run time.
    candidates = sorted(image_id for image_id in truth if truth[image_id])
    if arguments.limit and arguments.limit < len(candidates):
        step = (len(candidates) - 1) / (arguments.limit - 1)
        candidates = [candidates[round(index * step)] for index in range(arguments.limit)]

    model = YOLO(arguments.weights)
    model.to(arguments.device)
    names = model.names

    tally: collections.Counter = collections.Counter()
    for image_id in candidates:
        image_path = arguments.root / "val2017" / images[image_id]
        if not image_path.is_file():
            continue
        result = model.predict(
            source=str(image_path),
            imgsz=arguments.imgsz,
            conf=arguments.conf,
            verbose=False,
            device=arguments.device,
        )[0]
        predictions: list[tuple[str, np.ndarray, float]] = []
        for box, class_index, score in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            strict=True,
        ):
            product = product_class(str(names[int(class_index)]))
            if product is not None:
                predictions.append((product, box.astype(np.float64), float(score)))
        predictions.sort(key=lambda item: -item[2])

        remaining = list(truth[image_id])
        used: set[int] = set()
        for product, box, _ in predictions:
            best_index, best_overlap = -1, 0.0
            for index, (truth_product, truth_box) in enumerate(remaining):
                if index in used or truth_product != product:
                    continue
                overlap = box_iou(box, truth_box)
                if overlap > best_overlap:
                    best_index, best_overlap = index, overlap
            if best_overlap >= MATCH_IOU:
                used.add(best_index)
                tally[(product, "tp")] += 1
            else:
                tally[(product, "fp")] += 1
        for index, (truth_product, _) in enumerate(remaining):
            if index not in used:
                tally[(truth_product, "fn")] += 1

    rows = []
    print(f"{'class':<14}{'truth':>8}{'TP':>7}{'FP':>7}{'FN':>7}{'precision':>11}{'recall':>9}{'F1':>8}")
    for product in PRODUCT_CLASSES:
        tp = tally[(product, "tp")]
        fp = tally[(product, "fp")]
        fn = tally[(product, "fn")]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append(
            {
                "class": product,
                "truth_instances": tp + fn,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "f1": round(f1, 4),
            }
        )
        print(
            f"{product:<14}{tp + fn:>8}{tp:>7}{fp:>7}{fn:>7}"
            f"{precision:>11.4f}{recall:>9.4f}{f1:>8.4f}"
        )

    report = {
        "candidate": Path(arguments.weights).stem,
        "dataset": "COCO val2017 (general domain, the detector's own validation set)",
        "images_scored": len(candidates),
        "imgsz": arguments.imgsz,
        "conf": arguments.conf,
        "match_iou": MATCH_IOU,
        "caveat": (
            "Not parking-domain evidence: COCO is street photography and this is "
            "the detector's own validation split."
        ),
        "classes": rows,
    }
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
