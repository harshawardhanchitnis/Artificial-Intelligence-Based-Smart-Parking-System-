"""Reconstruct a fixed-camera sequence from an unseen camera, for development.

No prepared clip represents a camera the detector has never seen, so automatic
video calibration had nowhere to be tested where it matters most.  PKLot and
CNRPark do hold many chronological captures from one fixed camera on one day,
and putting them in time order reconstructs exactly what the calibration needs:
the same viewpoint, repeatedly, with the site's occupancy changing between
frames.

This is a **reconstructed time-lapse and not native footage**, and the
difference is not cosmetic.  Consecutive frames are minutes apart rather than
milliseconds, so a vehicle arrives or leaves between one frame and the next with
nothing in between.  Temporal smoothing therefore lags transitions here far more
than it would on real footage, and any figure measured on these sequences is a
pessimistic bound rather than a representative one.

Protocol: sources are drawn from the localization ``val_unseen`` split, which is
development data.  The protected holdout is never read.

    python ml/build_development_sequence.py --camera PKLot:UFPR04 --date UFPR04/2012-12-08
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

DEFAULT_MANIFEST = Path(
    "D:/Projects/AI Based Smart Parking System Data/prepared/v2-protocol/source-manifest.jsonl"
)
DEFAULT_OUTPUT = Path("D:/Projects/AI Based Smart Parking System Data/development/sequences")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--camera", required=True, help="e.g. PKLot:UFPR04")
    parser.add_argument("--date", required=True, help="group id, e.g. UFPR04/2012-12-08")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--name", default=None)
    arguments = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "backend"))
    from app.datasets.obb_dataset import assign_split, camera_identity

    rows = [
        json.loads(line)
        for line in arguments.manifest.read_text(encoding="utf-8").splitlines()
    ]
    selected = [
        row
        for row in rows
        if assign_split(row) == "val_unseen"
        and camera_identity(row) == arguments.camera
        and str(row["group_id"]) == arguments.date
    ]
    if len(selected) < 6:
        print(
            f"Only {len(selected)} unseen-camera frames match; "
            "a calibration sequence needs at least six."
        )
        return 1

    # Chronological order is the whole point: source ids end in a timestamp.
    selected.sort(key=lambda row: str(row["source_id"]))
    if len(selected) > arguments.frames:
        step = (len(selected) - 1) / (arguments.frames - 1)
        selected = [selected[round(index * step)] for index in range(arguments.frames)]

    data_root = arguments.manifest.parent
    first = cv2.imread(str(data_root / str(selected[0]["image_path"])))
    if first is None:
        print("The first frame could not be decoded")
        return 1
    height, width = first.shape[:2]

    name = arguments.name or arguments.date.replace("/", "-").lower()
    arguments.output.mkdir(parents=True, exist_ok=True)
    video_path = arguments.output / f"{name}.mp4"
    writer = cv2.VideoWriter(
        str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), arguments.fps, (width, height)
    )
    if not writer.isOpened():
        print("Video writer is unavailable")
        return 1

    frames: list[dict[str, object]] = []
    try:
        for row in selected:
            frame = cv2.imread(str(data_root / str(row["image_path"])))
            if frame is None:
                continue
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))
            writer.write(frame)
            labelled = [slot for slot in row["slots"] if slot.get("occupied") is not None]
            frames.append(
                {
                    "source_id": row["source_id"],
                    "condition": row.get("condition"),
                    "total_spaces": len(labelled),
                    "occupied_spaces": sum(1 for slot in labelled if slot["occupied"]),
                }
            )
    finally:
        writer.release()

    catalogue = {
        "id": name,
        "camera": arguments.camera,
        "group_id": arguments.date,
        "video_path": str(video_path),
        "frame_count": len(frames),
        "fps": arguments.fps,
        "continuity": "reconstructed time-lapse from chronological stills, NOT native footage",
        "split": "localization val_unseen (development; the protected holdout was not read)",
        "scientific_use": (
            "automatic multi-frame calibration on a camera absent from training; "
            "temporal figures are pessimistic because frames are minutes apart"
        ),
        "frames": frames,
    }
    (arguments.output / f"{name}.json").write_text(
        json.dumps(catalogue, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in catalogue.items() if k != "frames"}, indent=2))
    print(f"frames written: {len(frames)}  ->  {video_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
