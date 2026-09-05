from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.ml.localizer_training import (
    develop_fused_localizer,
    develop_hybrid_localizer,
    develop_localizer,
    finalize_localizer,
)


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Develop and freeze automatic slot localisation")
    parser.add_argument(
        "phase", choices=("develop", "evaluate-hybrid", "evaluate-fusion", "finalize")
    )
    parser.add_argument("--data-root", type=Path, default=settings.parking_data_root)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()
    model_root = args.model_root or args.data_root / "models"
    if args.phase == "develop":
        result = develop_localizer(args.data_root, model_root, epochs=args.epochs)
    elif args.phase == "evaluate-hybrid":
        result = develop_hybrid_localizer(args.data_root, model_root)
    elif args.phase == "evaluate-fusion":
        result = develop_fused_localizer(args.data_root, model_root)
    else:
        result = finalize_localizer(args.data_root, model_root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
