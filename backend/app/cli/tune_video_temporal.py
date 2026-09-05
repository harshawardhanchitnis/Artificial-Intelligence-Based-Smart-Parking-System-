from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.ml.temporal_training import tune_temporal_parameters


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Select fixed-camera temporal parameters")
    parser.add_argument("--data-root", type=Path, default=settings.parking_data_root)
    parser.add_argument("--model-root", type=Path)
    arguments = parser.parse_args()
    root = arguments.data_root.resolve()
    print(
        json.dumps(
            tune_temporal_parameters(root, arguments.model_root or root / "models"), indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
