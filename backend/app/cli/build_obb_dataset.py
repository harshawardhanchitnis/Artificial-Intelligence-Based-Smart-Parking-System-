"""Build the YOLO-OBB parking-space dataset from the prepared source manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.datasets.obb_dataset import (
    build_dataset,
    known_camera_leak,
    unseen_camera_overlap,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-mode", choices=("single", "occupancy"), default="single")
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    settings = get_settings()
    manifest = settings.parking_data_root / "prepared" / "v2-protocol" / "source-manifest.jsonl"
    if not manifest.is_file():
        print(f"Source manifest not found: {manifest}")
        return 1
    output = arguments.output or (
        settings.parking_data_root / "prepared" / f"obb-{arguments.label_mode}"
    )
    report = build_dataset(manifest, output, label_mode=arguments.label_mode)

    leak = unseen_camera_overlap(report)
    known_leak = known_camera_leak(report)
    print(json.dumps(report.as_dict()["images"], indent=2))
    print("instances:", json.dumps(report.as_dict()["instances"]))
    for split, cameras in report.cameras.items():
        print(f"  {split:12s} cameras={len(cameras)}")
    print("skipped polygons:", report.skipped_polygons)
    if leak:
        print(f"SPLIT LEAK (unseen camera also trained on): {sorted(leak)}")
        return 2
    if known_leak:
        print(f"SPLIT LEAK (val_known camera never trained on): {sorted(known_leak)}")
        return 2
    print("split integrity: unseen splits share no camera with training,")
    print("                 and every val_known camera is also a training camera")
    print("dataset written to", output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
