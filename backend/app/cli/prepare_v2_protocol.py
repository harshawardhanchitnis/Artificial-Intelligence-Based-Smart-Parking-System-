from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.datasets.v2_protocol import prepare_v2_protocol


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Prepare the leakage-safe V2 model corpus")
    parser.add_argument("--source-root", type=Path, default=settings.parking_data_root)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    args = parser.parse_args()
    report = prepare_v2_protocol(
        args.source_root,
        args.artifact_root,
        profile=args.profile,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
