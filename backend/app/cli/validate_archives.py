import argparse
from pathlib import Path

from app.core.config import get_settings
from app.datasets.archive_validator import validate_archive_catalogue


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Validate parking dataset archives")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=settings.parking_data_root,
        help="External data root containing the archives directory",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Open archive structures in addition to checking paths and sizes",
    )
    arguments = parser.parse_args()

    results = validate_archive_catalogue(arguments.data_root, deep=arguments.deep)
    print(f"Dataset root: {arguments.data_root}")
    print()
    for result in results:
        marker = "PASS" if result.valid else "FAIL"
        size_gb = result.size_bytes / (1024**3)
        path_text = str(result.path) if result.path else "not found"
        print(f"[{marker}] {result.label}")
        print(f"       {path_text}")
        print(f"       {size_gb:.2f} GiB - {result.message}")

    valid_count = sum(result.valid for result in results)
    print()
    print(f"Validated {valid_count}/{len(results)} required files.")
    return 0 if valid_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
