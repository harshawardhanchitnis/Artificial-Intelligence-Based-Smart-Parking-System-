from __future__ import annotations

import re
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import cv2

from app.datasets.integrity import atomic_json, read_jsonl, sha256_file
from app.datasets.preparation import PreparationError


def _timestamp(source_id: str) -> str:
    match = re.search(
        r"(\d{4}-\d{2}-\d{2})_(\d{2})[._-]?(\d{2})(?:[._-]?\d{2})?$",
        source_id,
    )
    return "".join(match.groups()) if match else source_id


def prepare_demo_videos(
    data_root: Path,
    *,
    videos_per_supported_dataset: int = 3,
    frames_per_video: int = 20,
    output_fps: float = 2.0,
) -> dict[str, object]:
    protocol_root = data_root / "prepared" / "v2-protocol"
    source_manifest = protocol_root / "source-manifest.jsonl"
    if not source_manifest.is_file():
        raise PreparationError("Prepare the standard V2 protocol before demo videos")
    sources = read_jsonl(source_manifest)
    by_group: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for source in sources:
        if source["dataset"] in {"PKLot", "CNRPark+EXT"}:
            by_group[(str(source["dataset"]), str(source["group_id"]))].append(source)

    destination_root = data_root / "demo" / "videos"
    destination_root.mkdir(parents=True, exist_ok=True)
    selected: list[dict[str, object]] = []
    for dataset in ("PKLot", "CNRPark+EXT"):
        candidates = sorted(
            (
                (key, rows)
                for key, rows in by_group.items()
                if key[0] == dataset and len(rows) >= min(8, frames_per_video)
            ),
            key=lambda item: (-len(item[1]), item[0][1]),
        )
        used_lots: set[str] = set()
        chosen = []
        for key, rows in candidates:
            lot = str(rows[0]["source_id"]).split("/", 1)[0]
            if lot in used_lots and len(used_lots) < 3:
                continue
            chosen.append((key, rows))
            used_lots.add(lot)
            if len(chosen) == videos_per_supported_dataset:
                break
        for index, ((_, group_id), rows) in enumerate(chosen, 1):
            ordered = sorted(rows, key=lambda row: _timestamp(str(row["source_id"])))
            if len(ordered) > frames_per_video:
                indices = [
                    round(value * (len(ordered) - 1) / (frames_per_video - 1))
                    for value in range(frames_per_video)
                ]
                ordered = [ordered[value] for value in indices]
            first = cv2.imread(str(protocol_root / str(ordered[0]["image_path"])))
            if first is None:
                continue
            height, width = first.shape[:2]
            filename = f"{dataset.lower().replace('+', '-').replace(' ', '-')}-{index}.mp4"
            destination = destination_root / filename
            writer = cv2.VideoWriter(
                str(destination), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height)
            )
            if not writer.isOpened():
                raise PreparationError("MP4 encoder is unavailable for prepared videos")
            written = []
            for row in ordered:
                frame = cv2.imread(str(protocol_root / str(row["image_path"])))
                if frame is None:
                    continue
                if frame.shape[1] != width or frame.shape[0] != height:
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
                written.append(
                    {
                        "source_id": row["source_id"],
                        "condition": row["condition"],
                        "occupied_spaces": sum(bool(slot["occupied"]) for slot in row["slots"]),
                        "total_spaces": len(row["slots"]),
                    }
                )
            writer.release()
            if written:
                selected.append(
                    {
                        "id": destination.stem,
                        "dataset": dataset,
                        "group_id": group_id,
                        "video_path": destination.relative_to(data_root).as_posix(),
                        "sha256": sha256_file(destination),
                        "frame_count": len(written),
                        "fps": output_fps,
                        "duration_seconds": round(len(written) / output_fps, 3),
                        "continuity": (
                            "same fixed camera/day, temporally ordered sampled time-lapse"
                        ),
                        "scientific_use": (
                            "temporal pipeline validation; not original-frame-rate footage"
                        ),
                        "frames": written,
                    }
                )
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "prepared_video_count": len(selected),
        "by_dataset": {
            dataset: sum(row["dataset"] == dataset for row in selected)
            for dataset in ("PKLot", "CNRPark+EXT", "ACPDS")
        },
        "acpds_status": (
            "No defensible continuous fixed-camera sequence is present; "
            "synthetic slideshows are not generated"
        ),
        "videos": selected,
    }
    atomic_json(destination_root / "catalogue.json", report)
    return report
