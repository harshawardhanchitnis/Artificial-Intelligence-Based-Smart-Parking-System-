from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.datasets.benchmark import verify_benchmark
from app.datasets.preparation import PreparationError
from app.ml.training import classification_metrics


def write_fixture(root: Path, *, overlap: bool = False) -> None:
    benchmark = root / "prepared" / "ml-benchmark"
    rows = []
    for partition_index, partition in enumerate(("train", "validation", "test")):
        for label in (0, 1):
            path = Path("patches") / partition / f"{label}.jpg"
            destination = benchmark / path
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = f"{partition}-{label}".encode()
            destination.write_bytes(payload)
            rows.append(
                {
                    "id": f"{partition}-{label}",
                    "dataset": "PKLot",
                    "partition": partition,
                    "group_id": "shared" if overlap and partition_index < 2 else partition,
                    "source_id": f"{partition}-{label}",
                    "label": label,
                    "patch_path": path.as_posix(),
                    "content_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    text = "".join(f"{json.dumps(row)}\n" for row in rows)
    (benchmark / "manifest.jsonl").write_text(text, encoding="utf-8")
    manifest_digest = hashlib.sha256((benchmark / "manifest.jsonl").read_bytes()).hexdigest()
    (benchmark / "report.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "manifest_sha256": manifest_digest,
            }
        ),
        encoding="utf-8",
    )


def test_benchmark_verifies_non_overlapping_groups(tmp_path: Path) -> None:
    write_fixture(tmp_path)
    report = verify_benchmark(tmp_path)
    assert report["valid"] is True
    assert report["sample_count"] == 6
    assert report["split_integrity"] is True


def test_benchmark_rejects_group_overlap(tmp_path: Path) -> None:
    write_fixture(tmp_path, overlap=True)
    with pytest.raises(PreparationError, match="group overlap"):
        verify_benchmark(tmp_path)


def test_scientific_metrics_include_balanced_accuracy_and_specificity() -> None:
    import numpy as np

    report = classification_metrics(np.asarray([0, 0, 1, 1]), np.asarray([0, 1, 1, 0]))
    assert report["accuracy"] == 0.5
    assert report["balanced_accuracy"] == 0.5
    assert report["specificity_vacant"] == 0.5
    assert report["confusion_matrix"]["false_vacant"] == 1
