from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.core.config import get_settings
from app.ml.training import TrainingError, train_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the local occupancy classifier")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    arguments = parser.parse_args()
    settings = get_settings()
    data_root = (arguments.data_root or settings.parking_data_root).resolve()
    model_root = (arguments.model_root or data_root / "models").resolve()
    try:
        report = train_model(
            data_root,
            model_root,
            random_state=arguments.random_state,
        )
    except (TrainingError, OSError, ValueError) as exc:
        print(f"Model training failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
