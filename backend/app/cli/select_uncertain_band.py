"""Select the occupancy uncertainty band from validation data only.

A binary decision forces every borderline space into vacant or occupied.  A
three-state presentation instead withholds a verdict where the calibrated
probability is genuinely ambiguous, so the states the interface *does* assert
are more reliable.

The band edges are chosen on the validation split.  The protected holdout is
never read here -- it stays reporting-only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.ml.occupancy_v3 import OccupancyV3Predictor, atomic_json


def load_split(manifest: Path, split: str) -> list[dict[str, object]]:
    rows = []
    with manifest.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("partition") == split:
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", default="validation", choices=("validation", "train"))
    parser.add_argument("--limit", type=int, default=15985)
    parser.add_argument("--batch", type=int, default=256)
    arguments = parser.parse_args()
    if arguments.split == "holdout":  # pragma: no cover - guarded by choices
        raise SystemExit("The protected holdout may not be used for selection")

    settings = get_settings()
    protocol = settings.parking_data_root / "prepared" / "v2-protocol"
    rows = load_split(protocol / "occupancy-manifest.jsonl", arguments.split)[: arguments.limit]
    print(f"scoring {len(rows):,} {arguments.split} crops")

    predictor = OccupancyV3Predictor.load(settings.model_root)
    probabilities: list[float] = []
    labels: list[int] = []
    for start in range(0, len(rows), arguments.batch):
        chunk = rows[start : start + arguments.batch]
        images = [Image.open(protocol / str(row["patch_path"])).convert("RGB") for row in chunk]
        values, _ = predictor.probabilities(images)
        for image in images:
            image.close()
        probabilities.extend(float(value) for value in values)
        labels.extend(int(row["label"]) for row in chunk)
        if start % (arguments.batch * 20) == 0:
            print(f"  {start + len(chunk):,}/{len(rows):,}", flush=True)

    probability = np.asarray(probabilities)
    label = np.asarray(labels)
    threshold = predictor.threshold

    best = None
    # Widen the band around the deployment threshold and keep the widest band
    # that still meets the accuracy target on what remains asserted.
    for lower in np.arange(threshold - 0.45, threshold, 0.01):
        for upper in np.arange(threshold + 0.01, threshold + 0.45, 0.01):
            if lower <= 0.02 or upper >= 0.98:
                continue
            confident = (probability < lower) | (probability > upper)
            coverage = float(confident.mean())
            if coverage < 0.90:
                continue
            asserted = probability[confident] > threshold
            truth = label[confident].astype(bool)
            positives = truth.sum()
            negatives = (~truth).sum()
            if not positives or not negatives:
                continue
            recall = float((asserted & truth).sum() / positives)
            specificity = float((~asserted & ~truth).sum() / negatives)
            balanced = (recall + specificity) / 2
            false_vacant = float((~asserted & truth).sum() / positives)
            score = balanced - 0.05 * (1 - coverage)
            if best is None or score > best["score"]:
                best = {
                    "score": score,
                    "lower": round(float(lower), 4),
                    "upper": round(float(upper), 4),
                    "coverage": round(coverage, 6),
                    "balanced_accuracy_on_asserted": round(balanced, 6),
                    "occupied_recall_on_asserted": round(recall, 6),
                    "vacant_specificity_on_asserted": round(specificity, 6),
                    "false_vacant_rate_on_asserted": round(false_vacant, 6),
                }

    baseline_asserted = probability > threshold
    baseline = {
        "balanced_accuracy": round(
            float(
                (
                    (baseline_asserted & label.astype(bool)).sum() / max(label.sum(), 1)
                    + ((~baseline_asserted) & (~label.astype(bool))).sum()
                    / max((~label.astype(bool)).sum(), 1)
                )
                / 2
            ),
            6,
        ),
        "false_vacant_rate": round(
            float(((~baseline_asserted) & label.astype(bool)).sum() / max(label.sum(), 1)), 6
        ),
    }

    report = {
        "selected_on": arguments.split,
        "samples": len(rows),
        "decision_threshold": threshold,
        "two_state_baseline": baseline,
        "uncertain_band": best,
        "policy": "band selected on validation only; protected holdout not read",
    }
    destination = settings.model_root / "development" / "occupancy-v3" / "uncertain-band.json"
    atomic_json(destination, report)
    print(json.dumps(report, indent=2))
    print("written", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
