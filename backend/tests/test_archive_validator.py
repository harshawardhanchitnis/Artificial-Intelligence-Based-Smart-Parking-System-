from pathlib import Path

from app.datasets.archive_validator import ARCHIVE_SPECS, validate_archive_catalogue


def test_catalogue_accepts_all_supported_filenames(tmp_path: Path) -> None:
    archives_root = tmp_path / "archives"
    for spec in ARCHIVE_SPECS:
        path = archives_root / spec.candidates[0]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")

    results = validate_archive_catalogue(tmp_path, enforce_minimum_size=False)
    assert len(results) == 7
    assert all(result.valid for result in results)


def test_catalogue_reports_missing_files(tmp_path: Path) -> None:
    results = validate_archive_catalogue(tmp_path)
    assert len(results) == 7
    assert not any(result.valid for result in results)
    assert all(result.message == "Required file is missing" for result in results)
