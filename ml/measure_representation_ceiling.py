"""Measure how much parking-space geometry a rotated rectangle cannot represent.

The training labels store the *true* four corners of each parking bay.  A
rotated-bounding-box detector cannot emit those corners: it regresses centre,
width, height and angle, so whatever it predicts is a rectangle.  Ultralytics
therefore fits a minimum-area rotated rectangle to the four corners before
training, and that fit is a ceiling no amount of training can lift.

This script measures that ceiling directly.  For every annotated bay it
compares the true quadrilateral against its own best-fit rotated rectangle and
reports the intersection-over-union, which is the highest score a perfect
rotated-box model could ever achieve on that bay.

Usage:
    python ml/measure_representation_ceiling.py --split val_unseen
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

DEFAULT_DATA = Path("D:/Projects/AI Based Smart Parking System Data/prepared/obb-single")


def polygon_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_area = float(cv2.contourArea(first))
    second_area = float(cv2.contourArea(second))
    if first_area <= 0 or second_area <= 0:
        return 0.0
    intersection, _ = cv2.intersectConvexConvex(first, second)
    union = first_area + second_area - float(intersection)
    return float(intersection) / union if union > 0 else 0.0


def corner_error(quad: np.ndarray, rect: np.ndarray) -> float:
    """Mean corner displacement, normalised by the bay's own scale.

    Dividing by the square root of the area makes the number comparable across
    bays that sit near the camera and bays far up the lot.
    """
    scale = max(float(np.sqrt(abs(cv2.contourArea(quad)))), 1e-6)
    # Match each true corner to the nearest rectangle corner under the cyclic
    # ordering that minimises total distance, so the metric is not an artefact
    # of which corner each representation happens to list first.
    best = None
    for shift in range(4):
        rolled = np.roll(rect, shift, axis=0)
        distance = float(np.mean(np.linalg.norm(quad - rolled, axis=1)))
        best = distance if best is None else min(best, distance)
    return (best or 0.0) / scale


def measure_split(data_root: Path, split: str) -> dict[str, object]:
    label_dir = data_root / "labels" / split
    image_dir = data_root / "images" / split
    ious: list[float] = []
    errors: list[float] = []
    images = 0
    for label_path in sorted(label_dir.glob("*.txt")):
        image_path = next((p for p in image_dir.glob(label_path.stem + ".*")), None)
        if image_path is None:
            continue
        with Image.open(image_path) as handle:
            width, height = handle.size
        images += 1
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) != 9:
                continue
            values = [float(v) for v in parts[1:]]
            quad = np.array(
                [[values[i] * width, values[i + 1] * height] for i in range(0, 8, 2)],
                dtype=np.float32,
            )
            if cv2.contourArea(quad) <= 0:
                quad = quad[::-1].copy()
            rect = cv2.boxPoints(cv2.minAreaRect(quad)).astype(np.float32)
            if cv2.contourArea(rect) <= 0:
                rect = rect[::-1].copy()
            ious.append(polygon_iou(quad, rect))
            errors.append(corner_error(quad, rect))
    if not ious:
        return {"split": split, "instances": 0}
    ordered = sorted(ious)
    return {
        "split": split,
        "images": images,
        "instances": len(ious),
        "mean_iou": round(statistics.fmean(ious), 6),
        "median_iou": round(statistics.median(ious), 6),
        "p05_iou": round(ordered[int(0.05 * len(ordered))], 6),
        "p25_iou": round(ordered[int(0.25 * len(ordered))], 6),
        "share_below_0_90": round(sum(1 for v in ious if v < 0.90) / len(ious), 6),
        "share_below_0_75": round(sum(1 for v in ious if v < 0.75) / len(ious), 6),
        "mean_corner_error_scaled": round(statistics.fmean(errors), 6),
        "p95_corner_error_scaled": round(sorted(errors)[int(0.95 * len(errors))], 6),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--splits", nargs="+", default=["train", "val_known", "val_unseen"])
    parser.add_argument("--out", type=Path, default=None)
    arguments = parser.parse_args()

    results = [measure_split(arguments.data, split) for split in arguments.splits]
    report = {"data": str(arguments.data), "splits": results}
    print(json.dumps(report, indent=2))
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
