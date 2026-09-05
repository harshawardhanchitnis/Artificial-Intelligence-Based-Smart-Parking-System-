from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.ml.scenario_regression import audit_prepared_scenarios


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare the frozen baseline and enhanced occupancy model on exposed scenarios"
    )
    parser.add_argument("--data-root", type=Path, default=get_settings().parking_data_root)
    parser.add_argument("--model-root", type=Path)
    arguments = parser.parse_args()
    root = arguments.data_root.resolve()
    output = root / "prepared" / "scenario-regression-report.json"
    report = audit_prepared_scenarios(
        root, (arguments.model_root or root / "models").resolve(), output_path=output
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
