from __future__ import annotations

import tarfile
import zipfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArchiveSpec:
    dataset: str
    label: str
    candidates: tuple[str, ...]
    minimum_bytes: int


@dataclass(frozen=True)
class ArchiveResult:
    dataset: str
    label: str
    path: Path | None
    exists: bool
    size_bytes: int
    valid: bool
    message: str

    def as_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "label": self.label,
            "path": str(self.path) if self.path else None,
            "exists": self.exists,
            "size_bytes": self.size_bytes,
            "size_gb": round(self.size_bytes / (1024**3), 3),
            "valid": self.valid,
            "message": self.message,
        }


ARCHIVE_SPECS = (
    ArchiveSpec("PKLot", "PKLot full archive", ("PKLot/PKLot.tar.gz",), 4_000_000_000),
    ArchiveSpec(
        "CNRPark+EXT",
        "CNR-EXT full images",
        ("CNRPark+EXT/CNR-EXT_FULL_IMAGE_1000x750.tar",),
        900_000_000,
    ),
    ArchiveSpec(
        "CNRPark+EXT",
        "CNR-EXT patches",
        ("CNRPark+EXT/CNR-EXT-Patches-150x150.zip",),
        350_000_000,
    ),
    ArchiveSpec(
        "CNRPark+EXT",
        "CNRPark patches",
        ("CNRPark+EXT/CNRPark-Patches-150x150.zip",),
        20_000_000,
    ),
    ArchiveSpec(
        "CNRPark+EXT",
        "CNRPark+EXT labels",
        ("CNRPark+EXT/CNRPark+EXT.csv",),
        10_000_000,
    ),
    ArchiveSpec(
        "CNRPark+EXT",
        "Official splits",
        ("CNRPark+EXT/splits.zip",),
        20_000_000,
    ),
    ArchiveSpec(
        "ACPDS",
        "ACPDS ROI archive",
        ("ACPDS/parking_rois_gopro.zip", "ACPDS/rois_gopro.zip"),
        300_000_000,
    ),
)


def _locate(archives_root: Path, candidates: tuple[str, ...]) -> Path | None:
    for relative_path in candidates:
        candidate = archives_root / Path(relative_path)
        if candidate.is_file():
            return candidate
    return None


def _inspect_archive(path: Path) -> tuple[bool, str]:
    suffixes = "".join(path.suffixes).lower()
    try:
        if suffixes.endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                if not archive.infolist():
                    return False, "ZIP archive is empty"
            return True, "ZIP structure opened successfully"
        if suffixes.endswith(".tar") or suffixes.endswith(".tar.gz"):
            with tarfile.open(path, mode="r:*") as archive:
                if archive.next() is None:
                    return False, "TAR archive is empty"
            return True, "TAR structure opened successfully"
        return True, "Regular data file found"
    except (OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        return False, f"Archive inspection failed: {error}"


def validate_archive_catalogue(
    data_root: Path,
    *,
    deep: bool = False,
    enforce_minimum_size: bool = True,
) -> list[ArchiveResult]:
    archives_root = data_root / "archives"
    results: list[ArchiveResult] = []

    for spec in ARCHIVE_SPECS:
        path = _locate(archives_root, spec.candidates)
        if path is None:
            results.append(
                ArchiveResult(
                    dataset=spec.dataset,
                    label=spec.label,
                    path=None,
                    exists=False,
                    size_bytes=0,
                    valid=False,
                    message="Required file is missing",
                )
            )
            continue

        size_bytes = path.stat().st_size
        if enforce_minimum_size and size_bytes < spec.minimum_bytes:
            results.append(
                ArchiveResult(
                    dataset=spec.dataset,
                    label=spec.label,
                    path=path,
                    exists=True,
                    size_bytes=size_bytes,
                    valid=False,
                    message=f"File is smaller than expected ({spec.minimum_bytes} byte minimum)",
                )
            )
            continue

        valid, message = _inspect_archive(path) if deep else (True, "File and size verified")
        results.append(
            ArchiveResult(
                dataset=spec.dataset,
                label=spec.label,
                path=path,
                exists=True,
                size_bytes=size_bytes,
                valid=valid,
                message=message,
            )
        )

    return results
