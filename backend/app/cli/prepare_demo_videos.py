from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.datasets.video_preparation import prepare_demo_videos


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Prepare traceable fixed-camera time-lapse videos")
    parser.add_argument("--data-root", type=Path, default=settings.parking_data_root)
    parser.add_argument("--videos-per-dataset", type=int, default=3)
    parser.add_argument("--frames", type=int, default=20)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare_demo_videos(
                args.data_root,
                videos_per_supported_dataset=args.videos_per_dataset,
                frames_per_video=args.frames,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
