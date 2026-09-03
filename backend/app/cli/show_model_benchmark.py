from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.core.config import get_settings
from app.ml.model_store import ModelNotReadyError, load_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Print the independent model benchmark")
    parser.add_argument("--model-root", type=Path, default=None)
    arguments = parser.parse_args()
    settings = get_settings()
    model_root = (arguments.model_root or settings.model_root).resolve()
    try:
        metadata = load_model(model_root).metadata
    except ModelNotReadyError as exc:
        print(f"Benchmark unavailable: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "model_name": metadata["model_name"],
                "trained_at": metadata["trained_at"],
                "fitting_policy": metadata["fitting_policy"],
                "training_samples": metadata["training_samples"],
                "validation_samples": metadata["validation_samples"],
                "test_samples": metadata["test_samples"],
                "decision_threshold": metadata["decision_threshold"],
                "benchmark_manifest_sha256": metadata["benchmark_manifest_sha256"],
                "independent_benchmark": metadata["independent_benchmark"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
