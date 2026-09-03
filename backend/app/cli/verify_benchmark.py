from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.core.config import get_settings
from app.datasets.benchmark import verify_benchmark
from app.datasets.preparation import PreparationError


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify the leakage-safe ML benchmark")
    parser.add_argument("--data-root", type=Path, default=None)
    arguments = parser.parse_args()
    data_root = (arguments.data_root or get_settings().parking_data_root).resolve()
    try:
        report = verify_benchmark(data_root)
    except (PreparationError, OSError, ValueError) as exc:
        print(f"Benchmark verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
