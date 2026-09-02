from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.core.config import get_settings
from app.datasets.preparation import PreparationError, verify_prepared_data


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the prepared demo catalogue")
    parser.add_argument("--data-root", type=Path, default=None)
    arguments = parser.parse_args()
    data_root = (arguments.data_root or get_settings().parking_data_root).resolve()
    try:
        result = verify_prepared_data(data_root)
    except PreparationError as exc:
        print(f"Prepared-data verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
