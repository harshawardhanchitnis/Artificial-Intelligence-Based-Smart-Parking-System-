from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.ml.training_v3 import develop_occupancy_v3, finalize_occupancy_v3


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Develop and freeze the enhanced occupancy model")
    parser.add_argument("phase", choices=("develop", "finalize"))
    parser.add_argument("--data-root", type=Path, default=settings.parking_data_root)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--profile", choices=("smoke", "standard"), default="standard")
    parser.add_argument("--epochs", type=int)
    args = parser.parse_args()
    model_root = args.model_root or args.data_root / "models"
    if args.phase == "develop":
        result = develop_occupancy_v3(
            args.data_root,
            model_root,
            profile=args.profile,
            epochs=args.epochs,
        )
    else:
        result = finalize_occupancy_v3(args.data_root, model_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
