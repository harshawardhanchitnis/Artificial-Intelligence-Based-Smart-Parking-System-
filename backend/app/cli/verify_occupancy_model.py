from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.core.config import get_settings
from app.ml.inference import verify_model
from app.ml.model_store import ModelNotReadyError


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify local inference on the catalogue")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--model-root", type=Path, default=None)
    arguments = parser.parse_args()
    settings = get_settings()
    data_root = (arguments.data_root or settings.parking_data_root).resolve()
    model_root = (arguments.model_root or data_root / "models").resolve()
    try:
        report = verify_model(data_root, model_root)
    except (ModelNotReadyError, OSError, ValueError) as exc:
        print(f"Model verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
