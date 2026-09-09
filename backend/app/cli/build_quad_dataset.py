"""Build a four-corner parking-space dataset from the prepared source manifest.

    python -m app.cli.build_quad_dataset --label-format pose
    python -m app.cli.build_quad_dataset --label-format seg
    python -m app.cli.build_quad_dataset --label-format obb --include-all-sources

The output directory encodes both axes of the experiment so the variants sit
side by side and can be trained against each other.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.datasets.obb_dataset import known_camera_leak, unseen_camera_overlap
from app.datasets.quad_dataset import LABEL_FORMATS, build_quad_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label-format", choices=LABEL_FORMATS, default="pose")
    parser.add_argument(
        "--include-all-sources",
        action="store_true",
        help="keep CNRPark+EXT, whose annotations are axis-aligned rather than bay outlines",
    )
    parser.add_argument("--output", type=Path, default=None)
    arguments = parser.parse_args()

    settings = get_settings()
    manifest = settings.parking_data_root / "prepared" / "v2-protocol" / "source-manifest.jsonl"
    if not manifest.is_file():
        print(f"Source manifest not found: {manifest}")
        return 1

    geometry_only = not arguments.include_all_sources
    suffix = "geom" if geometry_only else "all"
    output = arguments.output or (
        settings.parking_data_root / "prepared" / f"quad-{arguments.label_format}-{suffix}"
    )
    report = build_quad_dataset(
        manifest,
        output,
        label_format=arguments.label_format,
        geometry_sources_only=geometry_only,
    )

    print("images:   ", json.dumps(report.images))
    print("instances:", json.dumps(report.instances))
    for dataset, counts in sorted(report.datasets.items()):
        print(f"  {dataset:14s} {json.dumps(counts)}")
    for split, cameras in report.cameras.items():
        print(f"  {split:12s} cameras={len(cameras)}")
    print("skipped polygons:", report.skipped_polygons)

    leak = unseen_camera_overlap(report)
    known_leak = known_camera_leak(report)
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
