from __future__ import annotations

import itertools
import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from app.datasets.integrity import atomic_json, read_jsonl
from app.ml.geometry import rectify_slot
from app.ml.occupancy_v3 import OccupancyV3Predictor
from app.ml.temporal import TEMPORAL_CONFIG, TemporalParameters, TemporalState
from app.ml.training import classification_metrics


def _time_key(source_id: str) -> str:
    match = re.search(r"(\d{4}-\d{2}-\d{2})[_/](\d{2})[._-]?(\d{2})", source_id)
    return "".join(match.groups()) if match else source_id


def _validation_sequences(root: Path) -> list[list[dict[str, object]]]:
    rows = read_jsonl(root / "source-manifest.jsonl")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if row["dataset"] in {"PKLot", "CNRPark+EXT"} and row["partition"] == "validation":
            grouped[(str(row["dataset"]), str(row["group_id"]))].append(row)
    sequences = []
    for dataset in ("PKLot", "CNRPark+EXT"):
        candidates = sorted(
            (
                values
                for (name, _), values in grouped.items()
                if name == dataset and len(values) >= 8
            ),
            key=lambda values: (-len(values), str(values[0]["group_id"])),
        )[:2]
        for values in candidates:
            ordered = sorted(values, key=lambda row: _time_key(str(row["source_id"])))
            if len(ordered) > 12:
                indices = [round(index * (len(ordered) - 1) / 11) for index in range(12)]
                ordered = [ordered[index] for index in indices]
            if len({len(row["slots"]) for row in ordered}) == 1:
                sequences.append(ordered)
    if not sequences:
        raise RuntimeError("No fixed-camera validation sequences are available for temporal tuning")
    return sequences


def tune_temporal_parameters(data_root: Path, model_root: Path) -> dict[str, object]:
    protocol_root = data_root / "prepared" / "v2-protocol"
    predictor = OccupancyV3Predictor.load(model_root)
    sequences = _validation_sequences(protocol_root)
    prepared = []
    for rows in sequences:
        probability_frames = []
        label_frames = []
        for row in rows:
            with Image.open(protocol_root / str(row["image_path"])) as source:
                patches = [rectify_slot(source, slot["polygon"]) for slot in row["slots"]]
            probabilities, _ = predictor.probabilities(patches)
            probability_frames.append(probabilities)
            label_frames.append(np.asarray([bool(slot["occupied"]) for slot in row["slots"]]))
        prepared.append((rows[0]["dataset"], probability_frames, label_frames))

    candidates = []
    for median_window, ema_alpha, hysteresis, persistence in itertools.product(
        (1, 3, 5), (0.3, 0.5, 0.7), (0.04, 0.08, 0.12), (1, 2, 3)
    ):
        parameters = TemporalParameters(median_window, ema_alpha, hysteresis, persistence)
        by_dataset: dict[str, list[tuple[np.ndarray, np.ndarray]]] = defaultdict(list)
        transition_error = 0
        for dataset, probabilities, labels in prepared:
            state = TemporalState(len(probabilities[0]), predictor.threshold, parameters)
            predicted_frames = []
            for frame in probabilities:
                _, predicted, _ = state.update(frame)
                predicted_frames.append(predicted)
            predicted_array = np.asarray(predicted_frames)
            label_array = np.asarray(labels)
            by_dataset[str(dataset)].append((label_array, predicted_array))
            true_changes = int(np.sum(label_array[1:] != label_array[:-1]))
            predicted_changes = int(np.sum(predicted_array[1:] != predicted_array[:-1]))
            transition_error += abs(predicted_changes - true_changes)
        dataset_metrics = {}
        for dataset, arrays in by_dataset.items():
            truth = np.concatenate([pair[0].reshape(-1) for pair in arrays])
            predictions = np.concatenate([pair[1].reshape(-1) for pair in arrays])
            dataset_metrics[dataset] = classification_metrics(truth, predictions)
        worst_balanced = min(
            float(value["balanced_accuracy"]) for value in dataset_metrics.values()
        )
        false_vacant = sum(
            int(value["confusion_matrix"]["false_vacant"]) for value in dataset_metrics.values()
        )
        occupied = sum(
            int(value["confusion_matrix"]["true_occupied"])
            + int(value["confusion_matrix"]["false_vacant"])
            for value in dataset_metrics.values()
        )
        candidates.append(
            {
                "parameters": parameters.__dict__,
                "by_dataset": dataset_metrics,
                "worst_dataset_balanced_accuracy": round(worst_balanced, 6),
                "false_vacant_rate": round(false_vacant / max(occupied, 1), 6),
                "transition_count_error": transition_error,
            }
        )
    selected = max(
        candidates,
        key=lambda row: (
            float(row["worst_dataset_balanced_accuracy"]),
            -float(row["false_vacant_rate"]),
            -int(row["transition_count_error"]),
            -int(row["parameters"]["persistence"]),
        ),
    )
    report = {
        "schema_version": "1.0",
        "generated_at": datetime.now(UTC).isoformat(),
        "validation_locked": True,
        "selection_data": "occupancy-validation fixed-camera/day sequences only",
        "protected_holdout_used": False,
        "sequence_count": len(sequences),
        "candidate_count": len(candidates),
        "selected": selected["parameters"],
        "selected_metrics": {key: value for key, value in selected.items() if key != "parameters"},
        "all_candidates": candidates,
    }
    atomic_json(model_root / TEMPORAL_CONFIG, report)
    return report
