from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.core.config import get_settings
from app.datasets.scenario_qc import generate_scenario_qc


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate and record prepared-scenario overlay QC")
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument(
        "--confirm-reviewer",
        default=None,
        help="Reviewer name after visually checking every generated contact sheet",
    )
    arguments = parser.parse_args()
    root = (arguments.data_root or get_settings().parking_data_root).resolve()
    report = generate_scenario_qc(root, reviewer=arguments.confirm_reviewer)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
