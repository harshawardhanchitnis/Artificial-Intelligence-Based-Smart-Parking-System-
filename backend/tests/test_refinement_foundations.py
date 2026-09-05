from __future__ import annotations

import asyncio
import io
import json
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest
from fastapi import UploadFile
from PIL import Image

from app.core.config import Settings
from app.datasets.integrity import (
    quarantine_cross_partition_duplicates,
    read_jsonl,
    verify_partition_integrity,
    write_exposure_manifest,
)
from app.datasets.preparation import PreparationError
from app.datasets.video_preparation import _timestamp
from app.ml.geometry import order_polygon, rectify_slot, validate_polygon
from app.ml.temporal import TemporalParameters, TemporalState
from app.services.media_service import MediaValidationError, cleanup_media_artifacts, store_video


def test_perspective_rectification_masks_polygon_background() -> None:
    image = Image.new("RGB", (200, 120), "white")
    polygon = [[0.2, 0.2], [0.8, 0.1], [0.7, 0.9], [0.3, 0.8]]
    patch = rectify_slot(image, polygon, size=64)

    assert patch.size == (64, 64)
    assert validate_polygon(order_polygon(polygon)).valid is True
    assert np.asarray(patch).shape == (64, 64, 3)


def _manifest_row(identifier: str, partition: str, digest: str, perceptual: str) -> dict:
    return {
        "id": identifier,
        "dataset": "PKLot",
        "partition": partition,
        "group_id": f"group-{partition}",
        "source_id": f"source-{partition}",
        "content_sha256": digest,
        "perceptual_hash": perceptual,
    }


def test_duplicate_quarantine_prioritizes_protected_holdout() -> None:
    rows = [
        _manifest_row("train", "train", "same", "0000000000000000"),
        _manifest_row("train-unique", "train", "train-only", "3333333333333333"),
        _manifest_row("validation", "validation", "valid", "1111111111111111"),
        _manifest_row("holdout", "holdout", "same", "0000000000000000"),
    ]
    kept, quarantined = quarantine_cross_partition_duplicates(rows)

    assert {row["id"] for row in kept} == {"train-unique", "validation", "holdout"}
    assert quarantined[0]["id"] == "train"
    assert verify_partition_integrity(kept)["valid"] is True


def test_integrity_rejects_group_leakage() -> None:
    rows = [
        _manifest_row("train", "train", "a", "0000000000000000"),
        _manifest_row("validation", "validation", "b", "1111111111111111"),
        _manifest_row("holdout", "holdout", "c", "2222222222222222"),
    ]
    rows[1]["group_id"] = rows[0]["group_id"]
    with pytest.raises(PreparationError, match="Group leakage"):
        verify_partition_integrity(rows)


def test_exposure_manifest_unions_source_and_separate_artifact_catalogues(tmp_path) -> None:
    source_root = tmp_path / "source"
    artifact_root = tmp_path / "artifact"
    for root, identifier, color in (
        (source_root, "existing", "red"),
        (artifact_root, "expanded", "green"),
    ):
        media = root / "demo" / "media" / f"{identifier}.jpg"
        media.parent.mkdir(parents=True)
        Image.new("RGB", (200, 160), color).save(media)
        (root / "demo" / "catalogue.json").write_text(
            json.dumps(
                {
                    "scenarios": [
                        {
                            "id": identifier,
                            "dataset": "PKLot",
                            "image_path": media.relative_to(root).as_posix(),
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )

    report = write_exposure_manifest(source_root, artifact_root)
    rows = read_jsonl(artifact_root / "prepared" / "v2-protocol" / "historical-exposure.jsonl")

    assert report["row_count"] == 2
    assert {row["id"] for row in rows} == {"existing", "expanded"}
    assert all(len(str(row["source_sha256"])) == 64 for row in rows)


def test_temporal_state_requires_persistence_and_hysteresis() -> None:
    state = TemporalState(1, threshold=0.5, parameters=TemporalParameters())
    state.update(np.asarray([0.1]))
    _, first, events = state.update(np.asarray([0.95]))
    state.update(np.asarray([0.95]))
    _, second, events_second = state.update(np.asarray([0.95]))
    _, final, final_events = state.update(np.asarray([0.95]))

    assert first.tolist() == [False]
    assert events == []
    assert second.tolist() == [False]
    assert events_second == []
    assert final.tolist() == [True]
    assert final_events == [(0, False, True)]


def test_prepared_video_timestamp_uses_terminal_capture_time() -> None:
    morning = _timestamp("UFPR05/Sunny/2013-02-22/2013-02-22_06_15_00")
    evening = _timestamp("UFPR05/Cloudy/2013-02-22/2013-02-22_17_10_11")

    assert morning == "2013-02-220615"
    assert morning < evening


def test_video_magic_bytes_must_match_container(tmp_path) -> None:
    settings = Settings(parking_data_root=tmp_path)
    upload = UploadFile(filename="mislabelled.mp4", file=io.BytesIO(b"RIFF1234AVI invalid"))
    with pytest.raises(MediaValidationError, match="magic bytes"):
        asyncio.run(store_video(upload, settings))


def test_cleanup_preserves_referenced_media_and_removes_expired_orphan(tmp_path) -> None:
    settings = Settings(parking_data_root=tmp_path, media_retention_days=2)
    referenced = settings.media_root / "results" / "kept.jpg"
    orphan = settings.media_root / "results" / "orphan.jpg"
    referenced.parent.mkdir(parents=True)
    referenced.write_bytes(b"kept")
    orphan.write_bytes(b"orphan")
    old = (datetime.now(UTC) - timedelta(days=5)).timestamp()
    import os

    os.utime(referenced, (old, old))
    os.utime(orphan, (old, old))
    report = cleanup_media_artifacts(settings, {referenced.relative_to(tmp_path).as_posix()})

    assert referenced.exists()
    assert not orphan.exists()
    assert report["removed"] == 1
