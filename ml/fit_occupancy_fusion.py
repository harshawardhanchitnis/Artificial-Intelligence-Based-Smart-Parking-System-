"""Fit the occupancy fusion coefficients on development data.

The fusion model exists so that vehicle evidence can correct the crop
classifier where the classifier is wrong, without overriding it where it is
right.  Hand-chosen weights cannot be trusted to strike that balance, so the
weights are fitted: for every annotated bay in the occupancy *validation*
split, the classifier probability, the vehicle evidence and the association
strength are recorded alongside the verified label, and a logistic regression
recovers how much each signal is worth.

Two protocol points.  The fit reads the validation split only; the protected
holdout is never opened here, so it remains available to report a number this
fitting could not have influenced.  And the bay polygons come from the dataset
rather than from the space detector, because a bay the detector never proposed
has no label to fit against -- this isolates the question being asked, which is
how to weigh the signals, not how to find the bays.

    python ml/fit_occupancy_fusion.py --limit 320
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

DEFAULT_MANIFEST = Path(
    "D:/Projects/AI Based Smart Parking System Data/prepared/v2-protocol/source-manifest.jsonl"
)
DEFAULT_MODEL_ROOT = Path("D:/Projects/AI Based Smart Parking System Data/models")

# Order must match the coefficient names the runtime reads.
FEATURE_NAMES = (
    "occupancy_logit",
    "vehicle_present",
    "vehicle_absent",
    "slot_coverage",
    "vehicle_coverage",
)


def build_rows(
    manifest: Path, model_root: Path, limit: int | None, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.ml.association import associate
    from app.ml.geometry import rectify_slots
    from app.ml.occupancy_fusion import logit
    from app.ml.occupancy_v3 import OccupancyV3Predictor
    from app.ml.vehicle_detector import VehicleDetector

    classifier = OccupancyV3Predictor.load(model_root)
    detector = VehicleDetector.load(model_root)

    rows = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
    ]
    validation = [row for row in rows if row.get("partition") == "validation"]
    random.Random(seed).shuffle(validation)
    if limit:
        validation = validation[:limit]

    data_root = manifest.parent
    features: list[list[float]] = []
    labels: list[int] = []
    groups: list[str] = []
    counts = {"images": 0, "bays": 0, "with_vehicle": 0}

    for row in validation:
        image_path = data_root / str(row["image_path"])
        if not image_path.is_file():
            continue
        with Image.open(image_path) as handle:
            image = handle.convert("RGB")
        width, height = image.size

        slots = [slot for slot in row["slots"] if slot.get("occupied") is not None]
        if not slots:
            continue
        polygons = [slot["polygon"] for slot in slots]

        probabilities, _ = classifier.probabilities(rectify_slots(image, polygons))
        vehicles = detector.detect(image)
        association = associate(
            [[[x * width, y * height] for x, y in polygon] for polygon in polygons],
            [
                (
                    vehicle.box[0] * width,
                    vehicle.box[1] * height,
                    vehicle.box[2] * width,
                    vehicle.box[3] * height,
                )
                for vehicle in vehicles
            ],
        )

        counts["images"] += 1
        for index, (slot, probability) in enumerate(zip(slots, probabilities, strict=True)):
            occupying = association.occupying_links(index)
            best = max(occupying, key=lambda link: link.slot_coverage, default=None)
            present = best is not None
            confidence = vehicles[best.vehicle_index].confidence if best is not None else 0.0
            features.append(
                [
                    logit(float(probability)),
                    confidence if present else 0.0,
                    0.0 if present else 1.0,
                    best.slot_coverage if best is not None else 0.0,
                    best.vehicle_coverage if best is not None else 0.0,
                ]
            )
            labels.append(int(bool(slot["occupied"])))
            groups.append(f"{row['dataset']}|{row['group_id'].split('/', 1)[0]}")
            counts["bays"] += 1
            counts["with_vehicle"] += int(present)

    return (
        np.asarray(features, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(groups),
        counts,
    )


def select_band(
    probabilities: np.ndarray, labels: np.ndarray, target_error: float
) -> tuple[float, float]:
    """Narrowest band that keeps the error outside it below the target.

    Chosen on validation data.  The band is what makes "uncertain" mean
    something: outside it the system is asserting a verdict, so the error rate
    out there is the number that has to be controlled.

    Narrowest, not widest.  Widening a band can only remove borderline cases, so
    error outside it falls monotonically and *every* sufficiently wide band
    satisfies the constraint -- searching for the widest one therefore just
    returns the edge of the grid and abstains far more often than the error
    target requires.  The band the product wants is the smallest one that still
    holds the error down, because that is the one that answers most often.
    """
    best: tuple[float, float] | None = None
    for lower in np.arange(0.02, 0.50, 0.01):
        for upper in np.arange(0.51, 0.99, 0.01):
            decided = (probabilities <= lower) | (probabilities >= upper)
            if not decided.any():
                continue
            predicted = (probabilities[decided] >= upper).astype(int)
            error = float(np.mean(predicted != labels[decided]))
            if error > target_error:
                continue
            width = float(upper - lower)
            if best is None or width < (best[1] - best[0]):
                best = (float(lower), float(upper))
    # No band met the target: assert nothing rather than assert badly.
    return best if best is not None else (0.5, 0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-root", type=Path, default=DEFAULT_MODEL_ROOT)
    parser.add_argument("--limit", type=int, default=320)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--target-error", type=float, default=0.01)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="reuse extracted features so the fit can be re-run without re-inferring",
    )
    arguments = parser.parse_args()

    from sklearn.linear_model import LogisticRegression

    if arguments.cache and arguments.cache.is_file():
        cached = np.load(arguments.cache, allow_pickle=True)
        features, labels = cached["features"], cached["labels"]
        groups = cached["groups"]
        counts = dict(cached["counts"].item())
        print(f"reusing cached features from {arguments.cache}")
    else:
        features, labels, groups, counts = build_rows(
            arguments.manifest, arguments.model_root, arguments.limit, arguments.seed
        )
        if arguments.cache:
            arguments.cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                arguments.cache,
                features=features,
                labels=labels,
                groups=groups,
                counts=counts,
            )
    if features.size == 0:
        print("No development rows could be built")
        return 1

    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(features, labels)
    coefficients = {
        name: float(value) for name, value in zip(FEATURE_NAMES, model.coef_[0], strict=True)
    }
    coefficients["intercept"] = float(model.intercept_[0])
    coefficients["geometry_confidence"] = 0.0
    coefficients["prior_logit"] = 0.0

    fused = model.predict_proba(features)[:, 1]
    classifier_only = 1.0 / (1.0 + np.exp(-features[:, 0]))
    lower, upper = select_band(fused, labels, arguments.target_error)

    def report(probabilities: np.ndarray, name: str) -> dict[str, object]:
        decided = (probabilities <= lower) | (probabilities >= upper)
        predicted = (probabilities >= upper).astype(int)
        return {
            "signal": name,
            "balanced_accuracy_all": round(
                float(
                    (
                        np.mean(predicted[labels == 1] == 1)
                        + np.mean(predicted[labels == 0] == 0)
                    )
                    / 2
                ),
                6,
            ),
            "false_vacant": round(
                float(np.mean(predicted[labels == 1] == 0)), 6
            ),
            "false_occupied": round(
                float(np.mean(predicted[labels == 0] == 1)), 6
            ),
            "decided_share": round(float(np.mean(decided)), 6),
            "error_where_decided": round(
                float(np.mean(predicted[decided] != labels[decided])), 6
            ),
        }

    # Per camera, because an aggregate can hide a family the fusion hurts.
    per_group = []
    for name in sorted(set(groups.tolist())):
        mask = groups == name
        if mask.sum() < 50:
            continue
        entry: dict[str, object] = {"group": name, "bays": int(mask.sum())}
        for probabilities, key in ((classifier_only, "classifier"), (fused, "fused")):
            selected = probabilities[mask]
            selected_labels = labels[mask]
            decided = (selected <= lower) | (selected >= upper)
            predicted = (selected >= upper).astype(int)
            positives = selected_labels == 1
            negatives = selected_labels == 0
            entry[key] = {
                "balanced_accuracy": round(
                    float(
                        (
                            (np.mean(predicted[positives] == 1) if positives.any() else 0.0)
                            + (np.mean(predicted[negatives] == 0) if negatives.any() else 0.0)
                        )
                        / 2
                    ),
                    6,
                ),
                "false_vacant": round(
                    float(np.mean(predicted[positives] == 0)) if positives.any() else 0.0, 6
                ),
                "false_occupied": round(
                    float(np.mean(predicted[negatives] == 1)) if negatives.any() else 0.0, 6
                ),
                "asserted_coverage": round(float(np.mean(decided)), 6),
                "uncertain_rate": round(float(1 - np.mean(decided)), 6),
            }
        per_group.append(entry)

    development = {
        "fitted_on": "occupancy validation split (the protected holdout was not opened)",
        "per_camera": per_group,
        "images": counts["images"],
        "bays": counts["bays"],
        "bays_with_vehicle_evidence": counts["with_vehicle"],
        "target_error_outside_band": arguments.target_error,
        "comparison": [
            report(classifier_only, "classifier only"),
            report(fused, "fused with vehicle evidence"),
        ],
    }

    artifact = {
        "schema_version": "1.0",
        "model_name": "occupancy-fusion-v3",
        "coefficients": coefficients,
        "uncertain_band": {"lower": round(lower, 4), "upper": round(upper, 4)},
        "feature_order": list(FEATURE_NAMES),
        "fitted_on": development["fitted_on"],
        "trained_at": datetime.now(UTC).isoformat(),
        "development": development,
    }
    print(json.dumps(artifact, indent=2))
    destination = arguments.out or (arguments.model_root / "occupancy-fusion-v3.json")
    destination.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print("installed:", destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
