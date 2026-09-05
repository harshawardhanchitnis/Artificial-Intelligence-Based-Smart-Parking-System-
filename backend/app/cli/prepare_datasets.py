from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.core.config import get_settings
from app.datasets.benchmark import prepare_benchmark
from app.datasets.preparation import (
    PreparationError,
    plan_preparation,
    prepare_demo,
    prepare_full,
)


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description="Prepare approved parking datasets")
    command.add_argument("--data-root", type=Path, default=None)
    subcommands = command.add_subparsers(dest="profile", required=True)
    subcommands.add_parser("plan", help="Inspect files and disk requirements")
    demo = subcommands.add_parser("demo", help="Build the lightweight demo catalogue")
    demo.add_argument("--samples-per-dataset", type=int, default=10)
    demo.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Optional artifact root; source archives remain under --data-root",
    )
    demo.add_argument("--force", action="store_true")
    benchmark = subcommands.add_parser(
        "benchmark", help="Build leakage-safe ML train/validation/test patches"
    )
    benchmark.add_argument("--train-per-class", type=int, default=1000)
    benchmark.add_argument("--validation-per-class", type=int, default=300)
    benchmark.add_argument("--test-per-class", type=int, default=300)
    benchmark.add_argument("--random-seed", type=int, default=42)
    benchmark.add_argument("--force", action="store_true")
    full = subcommands.add_parser("full", help="Extract all source archives")
    full.add_argument("--confirm-full-extraction", action="store_true")
    full.add_argument("--force", action="store_true")
    return command


def main() -> int:
    arguments = parser().parse_args()
    data_root = (arguments.data_root or get_settings().parking_data_root).resolve()
    try:
        if arguments.profile == "plan":
            result = plan_preparation(data_root)
        elif arguments.profile == "demo":
            result = prepare_demo(
                data_root,
                output_root=arguments.output_root,
                samples_per_dataset=arguments.samples_per_dataset,
                force=arguments.force,
            )
        elif arguments.profile == "benchmark":
            result = prepare_benchmark(
                data_root,
                samples_per_class={
                    "train": arguments.train_per_class,
                    "validation": arguments.validation_per_class,
                    "test": arguments.test_per_class,
                },
                random_seed=arguments.random_seed,
                force=arguments.force,
            )
        else:
            result = prepare_full(
                data_root,
                confirmed=arguments.confirm_full_extraction,
                force=arguments.force,
            )
    except PreparationError as exc:
        print(f"Dataset preparation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
