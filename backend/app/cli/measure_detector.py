"""Measure the deployed detector the way the application actually runs it.

Ultralytics' own validation reports mAP over a sweep of confidences, which is
the right number for comparing checkpoints but not for choosing the threshold
the product ships with.  This walks the ONNX path on CPU -- the same code the
API calls -- and reports precision and recall per split at candidate operating
points, so the shipped threshold is chosen from measured behaviour.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from app.core.config import get_settings
from app.ml.generalized_localizer import GeneralizedDetector
from app.ml.geometry import polygon_iou

MATCH_IOU = 0.50


def load_truth(label_path: Path) -> list[list[list[float]]]:
    polygons = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 9:
            continue
        values = [float(value) for value in parts[1:9]]
        polygons.append([[values[i], values[i + 1]] for i in range(0, 8, 2)])
    return polygons


def score(
    detector: GeneralizedDetector,
    images: Path,
    labels: Path,
    confidence: float,
    limit: int,
) -> dict[str, object]:
    true_positive = false_positive = false_negative = 0
    elapsed = 0.0
    counted = 0
    for image_path in sorted(images.glob("*.jpg"))[:limit]:
        label_path = labels / f"{image_path.stem}.txt"
        if not label_path.is_file():
            continue
        truth = load_truth(label_path)
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
            started = time.perf_counter()
            detections = detector.detect(image, confidence=confidence)
            elapsed += time.perf_counter() - started
        counted += 1

        unmatched = list(truth)
        for detection in detections:
            best_index, best_iou = -1, 0.0
            for index, polygon in enumerate(unmatched):
                value = polygon_iou(detection["polygon"], polygon)
                if value > best_iou:
                    best_index, best_iou = index, value
            if best_iou >= MATCH_IOU:
                true_positive += 1
                unmatched.pop(best_index)
            else:
                false_positive += 1
        false_negative += len(unmatched)

    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    return {
        "confidence": confidence,
        "images": counted,
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(2 * precision * recall / max(precision + recall, 1e-9), 6),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "ms_per_image": round(elapsed / max(counted, 1) * 1000, 1),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("D:/Projects/AI Based Smart Parking System Data/prepared/obb-single"),
    )
    parser.add_argument("--splits", nargs="+", default=["val_known", "val_unseen"])
    parser.add_argument("--confidences", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.25])
    parser.add_argument("--limit", type=int, default=60)
    arguments = parser.parse_args()

    settings = get_settings()
    detector = GeneralizedDetector.load(settings.model_root)
    report: dict[str, object] = {"match_iou": MATCH_IOU, "limit_per_split": arguments.limit}

    for split in arguments.splits:
        rows = [
            score(
                detector,
                arguments.dataset / "images" / split,
                arguments.dataset / "labels" / split,
                confidence,
                arguments.limit,
            )
            for confidence in arguments.confidences
        ]
        report[split] = rows
        print(f"\n{split}")
        header = (
            f"  {'conf':>6} {'prec':>8} {'recall':>8} {'F1':>8} "
            f"{'TP':>6} {'FP':>6} {'FN':>6} {'ms':>7}"
        )
        print(header)
        for row in rows:
            print(
                f"  {row['confidence']:>6.2f} {row['precision']:>8.4f} {row['recall']:>8.4f} "
                f"{row['f1']:>8.4f} {row['true_positive']:>6d} "
                f"{row['false_positive']:>6d} {row['false_negative']:>6d} "
                f"{row['ms_per_image']:>7.0f}"
            )

    development = settings.model_root / "development" / "space-detector-v3"
    destination = development / "operating-points.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("\nwritten", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
