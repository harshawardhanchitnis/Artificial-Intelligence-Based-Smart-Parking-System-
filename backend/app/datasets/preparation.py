from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tarfile
import tempfile
import zipfile
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import BinaryIO
from xml.etree import ElementTree

import imagehash
from PIL import Image

from app.datasets.archive_validator import validate_archive_catalogue
from app.ml.geometry import order_polygon, validate_polygon

ProgressCallback = Callable[[str], None]
SCHEMA_VERSION = "2.0"
PKLOT_ARCHIVE = Path("archives/PKLot/PKLot.tar.gz")
CNR_FULL_ARCHIVE = Path("archives/CNRPark+EXT/CNR-EXT_FULL_IMAGE_1000x750.tar")
CNR_EXT_PATCHES = Path("archives/CNRPark+EXT/CNR-EXT-Patches-150x150.zip")
CNR_PARK_PATCHES = Path("archives/CNRPark+EXT/CNRPark-Patches-150x150.zip")
CNR_LABELS = Path("archives/CNRPark+EXT/CNRPark+EXT.csv")
CNR_SPLITS = Path("archives/CNRPark+EXT/splits.zip")
ACPDS_ARCHIVE_CANDIDATES = (
    Path("archives/ACPDS/parking_rois_gopro.zip"),
    Path("archives/ACPDS/rois_gopro.zip"),
)


class PreparationError(RuntimeError):
    """Raised when dataset preparation cannot safely continue."""


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _emit(callback: ProgressCallback | None, message: str) -> None:
    if callback:
        callback(message)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
        temporary_path = Path(handle.name)
    os.replace(temporary_path, path)


def _safe_relative_path(member_name: str) -> Path:
    normalized = member_name.replace("\\", "/")
    pure_path = PurePosixPath(normalized)
    if pure_path.is_absolute() or any(part in {"", ".", ".."} for part in pure_path.parts):
        raise PreparationError(f"Unsafe archive member path: {member_name}")
    return Path(*pure_path.parts)


def _safe_target(root: Path, member_name: str) -> Path:
    root_resolved = root.resolve()
    target = (root_resolved / _safe_relative_path(member_name)).resolve()
    if not target.is_relative_to(root_resolved):
        raise PreparationError(f"Archive member escapes extraction root: {member_name}")
    return target


def _write_stream(source: BinaryIO, destination: Path, *, force: bool) -> bool:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not force:
        return False
    temporary = destination.with_name(f".{destination.name}.partial")
    with temporary.open("wb") as output:
        shutil.copyfileobj(source, output, length=1024 * 1024)
    os.replace(temporary, destination)
    return True


def safe_extract_tar(
    archive_path: Path,
    destination: Path,
    *,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    written = 0
    skipped = 0
    destination.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, mode="r|*") as archive:
        for index, member in enumerate(archive, start=1):
            if member.issym() or member.islnk():
                raise PreparationError(f"Links are not allowed in archives: {member.name}")
            target = _safe_target(destination, member.name)
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                source = archive.extractfile(member)
                if source is None:
                    raise PreparationError(f"Could not read archive member: {member.name}")
                if _write_stream(source, target, force=force):
                    written += 1
                else:
                    skipped += 1
            if index % 25_000 == 0:
                _emit(progress, f"{archive_path.name}: scanned {index:,} members")
    return {"written": written, "skipped": skipped}


def safe_extract_zip(
    archive_path: Path,
    destination: Path,
    *,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    written = 0
    skipped = 0
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        for index, member in enumerate(archive.infolist(), start=1):
            unix_mode = member.external_attr >> 16
            if stat.S_IFMT(unix_mode) == stat.S_IFLNK:
                raise PreparationError(f"Links are not allowed in archives: {member.filename}")
            target = _safe_target(destination, member.filename)
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                with archive.open(member) as source:
                    if _write_stream(source, target, force=force):
                        written += 1
                    else:
                        skipped += 1
            if index % 25_000 == 0:
                _emit(progress, f"{archive_path.name}: scanned {index:,} members")
    return {"written": written, "skipped": skipped}


def _jpeg_dimensions(path: Path) -> tuple[int, int]:
    start_of_frame = {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
    with path.open("rb") as image:
        if image.read(2) != b"\xff\xd8":
            raise PreparationError(f"Not a JPEG image: {path}")
        while True:
            marker_start = image.read(1)
            if not marker_start:
                break
            if marker_start != b"\xff":
                continue
            marker = image.read(1)
            while marker == b"\xff":
                marker = image.read(1)
            if not marker:
                break
            marker_value = marker[0]
            if marker_value in {0xD8, 0xD9}:
                continue
            length_bytes = image.read(2)
            if len(length_bytes) != 2:
                break
            segment_length = int.from_bytes(length_bytes, "big")
            if segment_length < 2:
                break
            if marker_value in start_of_frame:
                payload = image.read(5)
                if len(payload) != 5:
                    break
                height = int.from_bytes(payload[1:3], "big")
                width = int.from_bytes(payload[3:5], "big")
                return width, height
            image.seek(segment_length - 2, 1)
    raise PreparationError(f"JPEG dimensions could not be read: {path}")


def _clamp(value: float) -> float:
    return round(min(1.0, max(0.0, value)), 6)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scenario(
    *,
    scenario_id: str,
    dataset: str,
    lot: str,
    condition: str,
    split: str | None,
    image_path: Path,
    data_root: Path,
    width: int,
    height: int,
    slots: list[dict[str, object]],
) -> dict[str, object]:
    normalized_slots = []
    for slot in slots:
        polygon = order_polygon(slot["polygon"])
        quality = validate_polygon(polygon)
        if not quality.valid:
            raise PreparationError(
                f"Invalid geometry in {scenario_id} slot {slot['id']}: {quality.reason}"
            )
        normalized_slots.append(
            {
                **slot,
                "polygon": [[round(x, 6), round(y, 6)] for x, y in polygon],
                "geometry_area": round(quality.area, 8),
            }
        )
    slots = normalized_slots
    occupied = sum(bool(slot["occupied"]) for slot in slots)
    total = len(slots)
    with Image.open(image_path) as image:
        perceptual_hash = str(imagehash.phash(image.convert("RGB")))
    return {
        "schema_version": SCHEMA_VERSION,
        "id": scenario_id,
        "dataset": dataset,
        "lot": lot,
        "condition": condition,
        "split": split,
        "image_path": image_path.relative_to(data_root).as_posix(),
        "image_width": width,
        "image_height": height,
        "total_spaces": total,
        "occupied_spaces": occupied,
        "vacant_spaces": total - occupied,
        "annotation_source": "dataset_ground_truth",
        "source_sha256": _sha256_file(image_path),
        "perceptual_hash": perceptual_hash,
        "evaluation_role": "exposed_scenario_regression",
        "geometry_qc": "passed",
        "slots": slots,
    }


def _parse_pklot_xml(xml_path: Path, width: int, height: int) -> list[dict[str, object]]:
    root = ElementTree.parse(xml_path).getroot()
    slots: list[dict[str, object]] = []
    for space in root.findall("space"):
        points = []
        contour = space.find("contour")
        if contour is None:
            continue
        for point in contour.findall("point"):
            points.append(
                [
                    _clamp(float(point.attrib["x"]) / width),
                    _clamp(float(point.attrib["y"]) / height),
                ]
            )
        if len(points) < 3:
            continue
        slots.append(
            {
                "id": str(space.attrib.get("id", len(slots) + 1)),
                "polygon": points,
                "occupied": space.attrib.get("occupied") == "1",
            }
        )
    return slots


def _diverse_scenario_subset(
    scenarios: list[dict[str, object]], sample_count: int
) -> list[dict[str, object]]:
    """Cover conditions and locations before balancing remaining selections."""

    ordered = sorted(scenarios, key=lambda row: str(row["id"]))
    selected: list[dict[str, object]] = []
    selected_ids: set[str] = set()
    used_lots: set[str] = set()

    def choose(row: dict[str, object]) -> None:
        selected.append(row)
        selected_ids.add(str(row["id"]))
        used_lots.add(str(row["lot"]))

    for condition in sorted({str(row["condition"]) for row in ordered}):
        candidates = [row for row in ordered if str(row["condition"]) == condition]
        choose(min(candidates, key=lambda row: (str(row["lot"]) in used_lots, str(row["id"]))))
        if len(selected) == sample_count:
            return selected
    for lot in sorted({str(row["lot"]) for row in ordered}):
        if lot in used_lots:
            continue
        choose(next(row for row in ordered if str(row["lot"]) == lot))
        if len(selected) == sample_count:
            return selected
    while len(selected) < sample_count:
        remaining = [row for row in ordered if str(row["id"]) not in selected_ids]
        if not remaining:
            break
        lot_counts = {lot: sum(str(row["lot"]) == lot for row in selected) for lot in used_lots}
        condition_counts = {
            condition: sum(str(row["condition"]) == condition for row in selected)
            for condition in {str(row["condition"]) for row in ordered}
        }
        choose(
            min(
                remaining,
                key=lambda row: (
                    lot_counts.get(str(row["lot"]), 0),
                    condition_counts[str(row["condition"])],
                    str(row["id"]),
                ),
            )
        )
    return selected


def _prepare_pklot_demo(
    source_root: Path,
    output_root: Path,
    sample_count: int,
    *,
    force: bool,
    progress: ProgressCallback | None,
) -> list[dict[str, object]]:
    archive_path = source_root / PKLOT_ARCHIVE
    media_root = output_root / "demo" / "media" / "pklot"
    metadata_root = output_root / "demo" / "metadata" / "pklot"
    selected: dict[tuple[str, str, str], dict[str, object]] = {}
    group_counts: dict[tuple[str, str], int] = defaultdict(int)
    group_limit = max(1, math.ceil(sample_count / 9))

    _emit(progress, "PKLot: scanning archive for curated full-image/XML pairs")
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for index, member in enumerate(archive, start=1):
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) != 6 or parts[:2] != ("PKLot", "PKLot"):
                continue
            lot, weather, _, filename = parts[2], parts[3], parts[4], parts[5]
            suffix = Path(filename).suffix.lower()
            if suffix not in {".jpg", ".xml"}:
                continue
            stem = Path(filename).stem
            key = (lot, weather, stem)
            group = (lot, weather)
            if key not in selected:
                if group_counts[group] >= group_limit:
                    continue
                scenario_id = f"pklot-{_slug(lot)}-{_slug(weather)}-{_slug(stem)}"
                selected[key] = {"id": scenario_id, "lot": lot, "condition": weather}
                group_counts[group] += 1

            record = selected.get(key)
            if record is None:
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            if suffix == ".jpg":
                destination = media_root / f"{record['id']}.jpg"
                _write_stream(source, destination, force=force)
                record["image"] = destination
            else:
                destination = metadata_root / f"{record['id']}.xml"
                _write_stream(source, destination, force=force)
                record["xml"] = destination
            if index % 100_000 == 0:
                _emit(progress, f"PKLot: scanned {index:,} archive members")

    scenarios = []
    for record in selected.values():
        image_path = record.get("image")
        xml_path = record.get("xml")
        if not isinstance(image_path, Path) or not isinstance(xml_path, Path):
            continue
        width, height = _jpeg_dimensions(image_path)
        slots = _parse_pklot_xml(xml_path, width, height)
        if slots:
            scenarios.append(
                _scenario(
                    scenario_id=str(record["id"]),
                    dataset="PKLot",
                    lot=str(record["lot"]),
                    condition=str(record["condition"]),
                    split=None,
                    image_path=image_path,
                    data_root=output_root,
                    width=width,
                    height=height,
                    slots=slots,
                )
            )
    scenarios = _diverse_scenario_subset(scenarios, sample_count)
    _emit(progress, f"PKLot: prepared {len(scenarios)} scenarios")
    return scenarios


def _read_cnr_geometry(metadata_root: Path) -> dict[str, dict[str, tuple[float, ...]]]:
    geometry: dict[str, dict[str, tuple[float, ...]]] = {}
    for camera_csv in metadata_root.glob("camera*.csv"):
        camera = camera_csv.stem.lower()
        geometry[camera] = {}
        with camera_csv.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                geometry[camera][str(row["SlotId"])] = (
                    float(row["X"]),
                    float(row["Y"]),
                    float(row["W"]),
                    float(row["H"]),
                )
    return geometry


def _cnr_datetime_to_stem(value: str) -> str:
    date, time = value.split("_", maxsplit=1)
    return f"{date}_{time.replace('.', '')}"


def _prepare_cnr_demo(
    source_root: Path,
    output_root: Path,
    sample_count: int,
    *,
    force: bool,
    progress: ProgressCallback | None,
) -> list[dict[str, object]]:
    archive_path = source_root / CNR_FULL_ARCHIVE
    media_root = output_root / "demo" / "media" / "cnrpark-ext"
    metadata_root = output_root / "demo" / "metadata" / "cnrpark-ext"
    selected: dict[tuple[str, str], dict[str, object]] = {}
    group_counts: dict[tuple[str, str], int] = defaultdict(int)
    group_limit = max(1, math.ceil(sample_count / 27))

    _emit(progress, "CNRPark+EXT: selecting full images and camera geometry")
    with tarfile.open(archive_path, mode="r|") as archive:
        for member in archive:
            if not member.isfile():
                continue
            name = PurePosixPath(member.name)
            if name.name.lower().startswith("camera") and name.suffix.lower() == ".csv":
                source = archive.extractfile(member)
                if source:
                    _write_stream(source, metadata_root / name.name.lower(), force=force)
                continue
            parts = name.parts
            if len(parts) != 5 or parts[0] != "FULL_IMAGE_1000x750":
                continue
            weather, camera, filename = parts[1], parts[3].lower(), parts[4]
            if Path(filename).suffix.lower() not in {".jpg", ".jpeg"}:
                continue
            stem = Path(filename).stem
            key = (camera, stem)
            group = (weather, camera)
            if key in selected:
                continue
            if group_counts[group] >= group_limit:
                continue
            scenario_id = f"cnrpark-ext-{_slug(camera)}-{_slug(weather)}-{_slug(stem)}"
            destination = media_root / f"{scenario_id}.jpg"
            source = archive.extractfile(member)
            if source is None:
                continue
            _write_stream(source, destination, force=force)
            selected[key] = {
                "id": scenario_id,
                "camera": camera,
                "condition": weather.title(),
                "image": destination,
            }
            group_counts[group] += 1

    occupancy: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    with (source_root / CNR_LABELS).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            camera_value = str(row.get("camera", ""))
            if not camera_value.isdigit():
                continue
            camera = f"camera{int(camera_value)}"
            try:
                stem = _cnr_datetime_to_stem(str(row["datetime"]))
            except (KeyError, ValueError):
                continue
            key = (camera, stem)
            if key in selected:
                occupancy[key][str(row["slot_id"])] = str(row["occupancy"]) == "1"

    geometry = _read_cnr_geometry(metadata_root)
    scenarios = []
    original_width, original_height = 2592.0, 1944.0
    for key, record in selected.items():
        camera = str(record["camera"])
        slot_states = occupancy.get(key, {})
        slots = []
        for slot_id, state in slot_states.items():
            rectangle = geometry.get(camera, {}).get(slot_id)
            if rectangle is None:
                continue
            x, y, width, height = rectangle
            polygon = [
                [_clamp(x / original_width), _clamp(y / original_height)],
                [_clamp((x + width) / original_width), _clamp(y / original_height)],
                [
                    _clamp((x + width) / original_width),
                    _clamp((y + height) / original_height),
                ],
                [_clamp(x / original_width), _clamp((y + height) / original_height)],
            ]
            slots.append({"id": slot_id, "polygon": polygon, "occupied": state})
        image_path = Path(str(record["image"]))
        width, height = _jpeg_dimensions(image_path)
        if slots:
            scenarios.append(
                _scenario(
                    scenario_id=str(record["id"]),
                    dataset="CNRPark+EXT",
                    lot=camera,
                    condition=str(record["condition"]),
                    split=None,
                    image_path=image_path,
                    data_root=output_root,
                    width=width,
                    height=height,
                    slots=slots,
                )
            )
    scenarios = _diverse_scenario_subset(scenarios, sample_count)
    _emit(progress, f"CNRPark+EXT: prepared {len(scenarios)} scenarios")
    return scenarios


def _acpds_archive(data_root: Path) -> Path:
    for relative_path in ACPDS_ARCHIVE_CANDIDATES:
        candidate = data_root / relative_path
        if candidate.is_file():
            return candidate
    raise PreparationError("ACPDS archive is missing")


def _prepare_acpds_demo(
    source_root: Path,
    output_root: Path,
    sample_count: int,
    *,
    force: bool,
    progress: ProgressCallback | None,
) -> list[dict[str, object]]:
    archive_path = _acpds_archive(source_root)
    media_root = output_root / "demo" / "media" / "acpds"
    with zipfile.ZipFile(archive_path) as archive:
        with archive.open("annotations.json") as handle:
            annotations = json.load(handle)

        selected: list[tuple[str, int]] = []
        allocation = {"train": 4, "valid": 3, "test": 3}
        if sample_count != 10:
            allocation = {
                split: sample_count // 3 + int(index < sample_count % 3)
                for index, split in enumerate(("train", "valid", "test"))
            }
        for split in ("train", "valid", "test"):
            file_names = annotations[split]["file_names"]
            count = min(allocation[split], len(file_names))
            if count == 1:
                indices = [len(file_names) // 2]
            else:
                indices = [
                    round(index * (len(file_names) - 1) / (count - 1)) for index in range(count)
                ]
            selected.extend((split, int(index)) for index in indices)

        scenarios = []
        for split, index in selected:
            split_data = annotations[split]
            filename = str(split_data["file_names"][index])
            scenario_id = f"acpds-{split}-{_slug(Path(filename).stem)}"
            image_path = media_root / f"{scenario_id}.jpg"
            with archive.open(f"images/{filename}") as source:
                _write_stream(source, image_path, force=force)
            width, height = _jpeg_dimensions(image_path)
            rois = split_data["rois_list"][index]
            states = split_data["occupancy_list"][index]
            slots = [
                {
                    "id": str(slot_index + 1),
                    "polygon": [[_clamp(float(x)), _clamp(float(y))] for x, y in polygon],
                    "occupied": bool(states[slot_index]),
                }
                for slot_index, polygon in enumerate(rois)
            ]
            scenarios.append(
                _scenario(
                    scenario_id=scenario_id,
                    dataset="ACPDS",
                    lot=f"{split.title()} split",
                    condition="Mixed",
                    split=split,
                    image_path=image_path,
                    data_root=output_root,
                    width=width,
                    height=height,
                    slots=slots,
                )
            )
    _emit(progress, f"ACPDS: prepared {len(scenarios)} scenarios")
    return scenarios


def _catalogue(
    scenarios: list[dict[str, object]], profile: str, required_minimum_per_dataset: int
) -> dict[str, object]:
    datasets = []
    for dataset_name in ("PKLot", "CNRPark+EXT", "ACPDS"):
        dataset_scenarios = [row for row in scenarios if row["dataset"] == dataset_name]
        datasets.append(
            {
                "name": dataset_name,
                "scenario_count": len(dataset_scenarios),
                "lots": sorted({str(row["lot"]) for row in dataset_scenarios}),
                "conditions": sorted({str(row["condition"]) for row in dataset_scenarios}),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "profile": profile,
        "required_minimum_per_dataset": required_minimum_per_dataset,
        "scenario_count": len(scenarios),
        "datasets": datasets,
        "scenarios": scenarios,
    }


def prepare_demo(
    data_root: Path,
    *,
    output_root: Path | None = None,
    samples_per_dataset: int = 10,
    force: bool = False,
    progress: ProgressCallback | None = print,
    validate_sizes: bool = True,
) -> dict[str, object]:
    if samples_per_dataset < 1 or samples_per_dataset > 30:
        raise PreparationError("samples_per_dataset must be between 1 and 30")
    data_root = data_root.resolve()
    output_root = (output_root or data_root).resolve()
    validation = validate_archive_catalogue(data_root, enforce_minimum_size=validate_sizes)
    invalid = [result.label for result in validation if not result.valid]
    if invalid:
        raise PreparationError(f"Archive validation failed: {', '.join(invalid)}")

    scenarios = []
    scenarios.extend(
        _prepare_pklot_demo(
            data_root,
            output_root,
            samples_per_dataset,
            force=force,
            progress=progress,
        )
    )
    scenarios.extend(
        _prepare_cnr_demo(
            data_root,
            output_root,
            samples_per_dataset,
            force=force,
            progress=progress,
        )
    )
    scenarios.extend(
        _prepare_acpds_demo(
            data_root,
            output_root,
            samples_per_dataset,
            force=force,
            progress=progress,
        )
    )
    scenarios.sort(key=lambda row: (str(row["dataset"]), str(row["id"])))
    if not scenarios:
        raise PreparationError("No scenarios were produced")

    prepared_root = output_root / "prepared"
    for dataset_name, file_name in (
        ("PKLot", "pklot.jsonl"),
        ("CNRPark+EXT", "cnrpark_ext.jsonl"),
        ("ACPDS", "acpds.jsonl"),
    ):
        _atomic_jsonl(
            prepared_root / "manifests" / file_name,
            [row for row in scenarios if row["dataset"] == dataset_name],
        )

    catalogue = _catalogue(scenarios, "demo", min(10, samples_per_dataset))
    catalogue_path = output_root / "demo" / "catalogue.json"
    _atomic_json(catalogue_path, catalogue)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "profile": "demo",
        "samples_per_dataset_requested": samples_per_dataset,
        "scenario_count": len(scenarios),
        "catalogue_path": catalogue_path.relative_to(output_root).as_posix(),
        "dataset_counts": {
            dataset["name"]: dataset["scenario_count"] for dataset in catalogue["datasets"]
        },
    }
    _atomic_json(prepared_root / "preparation-report.json", report)
    return report


def plan_preparation(data_root: Path) -> dict[str, object]:
    data_root = data_root.resolve()
    archives = []
    relative_paths = (
        PKLOT_ARCHIVE,
        CNR_FULL_ARCHIVE,
        CNR_EXT_PATCHES,
        CNR_PARK_PATCHES,
        CNR_LABELS,
        CNR_SPLITS,
        next(
            (path for path in ACPDS_ARCHIVE_CANDIDATES if (data_root / path).is_file()),
            ACPDS_ARCHIVE_CANDIDATES[0],
        ),
    )
    for relative_path in relative_paths:
        path = data_root / relative_path
        archives.append(
            {
                "path": relative_path.as_posix(),
                "exists": path.is_file(),
                "size_bytes": path.stat().st_size if path.is_file() else 0,
            }
        )
    archive_bytes = sum(int(row["size_bytes"]) for row in archives)
    free_bytes = shutil.disk_usage(data_root).free
    return {
        "data_root": str(data_root),
        "archives": archives,
        "archive_bytes": archive_bytes,
        "free_bytes": free_bytes,
        "estimated_full_extraction_bytes": archive_bytes * 3,
        "demo_profile": "Curated full images and normalized annotations only",
        "full_profile": "All archives extracted into raw; requires explicit confirmation",
    }


def prepare_full(
    data_root: Path,
    *,
    confirmed: bool,
    force: bool = False,
    progress: ProgressCallback | None = print,
) -> dict[str, object]:
    if not confirmed:
        raise PreparationError("Full extraction requires explicit confirmation")
    data_root = data_root.resolve()
    plan = plan_preparation(data_root)
    missing = [row["path"] for row in plan["archives"] if not row["exists"]]
    if missing:
        raise PreparationError(f"Missing archives: {', '.join(str(item) for item in missing)}")
    if int(plan["free_bytes"]) < int(plan["estimated_full_extraction_bytes"]):
        raise PreparationError("Insufficient free disk space for guarded full extraction estimate")

    raw_root = data_root / "raw"
    operations = [
        ("PKLot", data_root / PKLOT_ARCHIVE, raw_root / "PKLot", "tar"),
        ("CNR full images", data_root / CNR_FULL_ARCHIVE, raw_root / "CNRPark+EXT", "tar"),
        ("CNR-EXT patches", data_root / CNR_EXT_PATCHES, raw_root / "CNRPark+EXT", "zip"),
        ("CNRPark patches", data_root / CNR_PARK_PATCHES, raw_root / "CNRPark+EXT", "zip"),
        ("CNR splits", data_root / CNR_SPLITS, raw_root / "CNRPark+EXT", "zip"),
        ("ACPDS", _acpds_archive(data_root), raw_root / "ACPDS", "zip"),
    ]
    results = []
    for label, archive_path, destination, archive_type in operations:
        _emit(progress, f"Extracting {label} to {destination}")
        if archive_type == "tar":
            result = safe_extract_tar(archive_path, destination, force=force, progress=progress)
        else:
            result = safe_extract_zip(archive_path, destination, force=force, progress=progress)
        results.append({"label": label, "archive": str(archive_path), **result})

    labels_destination = raw_root / "CNRPark+EXT" / "CNRPark+EXT.csv"
    labels_destination.parent.mkdir(parents=True, exist_ok=True)
    if force or not labels_destination.exists():
        shutil.copy2(data_root / CNR_LABELS, labels_destination)

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_at": _now(),
        "profile": "full",
        "operations": results,
    }
    _atomic_json(data_root / "prepared" / "full-extraction-report.json", report)
    return report


def verify_prepared_data(data_root: Path) -> dict[str, object]:
    data_root = data_root.resolve()
    catalogue_path = data_root / "demo" / "catalogue.json"
    if not catalogue_path.is_file():
        raise PreparationError("Demo catalogue is missing; run demo preparation first")
    with catalogue_path.open(encoding="utf-8") as handle:
        catalogue = json.load(handle)
    if catalogue.get("schema_version") != SCHEMA_VERSION:
        raise PreparationError("Unsupported catalogue schema")
    scenarios = catalogue.get("scenarios")
    if not isinstance(scenarios, list) or not scenarios:
        raise PreparationError("Catalogue contains no scenarios")

    dataset_counts: dict[str, int] = defaultdict(int)
    source_hashes: set[str] = set()
    perceptual_hashes: list[tuple[str, str]] = []
    checked_slots = 0
    for scenario in scenarios:
        image_path = (data_root / str(scenario["image_path"])).resolve()
        if not image_path.is_relative_to(data_root) or not image_path.is_file():
            raise PreparationError(f"Scenario image is missing or unsafe: {scenario['id']}")
        slots = scenario.get("slots")
        if not isinstance(slots, list) or not slots:
            raise PreparationError(f"Scenario has no slots: {scenario['id']}")
        occupied = sum(bool(slot["occupied"]) for slot in slots)
        if occupied != int(scenario["occupied_spaces"]):
            raise PreparationError(f"Occupied total mismatch: {scenario['id']}")
        if len(slots) != int(scenario["total_spaces"]):
            raise PreparationError(f"Slot total mismatch: {scenario['id']}")
        source_hash = str(scenario.get("source_sha256", ""))
        if len(source_hash) != 64:
            raise PreparationError(f"Missing source hash: {scenario['id']}")
        if source_hash in source_hashes:
            raise PreparationError(f"Exact duplicate scenario image: {scenario['id']}")
        source_hashes.add(source_hash)
        perceptual_hash = str(scenario.get("perceptual_hash", ""))
        if len(perceptual_hash) != 16:
            raise PreparationError(f"Missing perceptual hash: {scenario['id']}")
        for other_id, other_hash in perceptual_hashes:
            distance = imagehash.hex_to_hash(perceptual_hash) - imagehash.hex_to_hash(other_hash)
            if distance <= 1:
                raise PreparationError(
                    f"Near-duplicate scenarios detected: {other_id} and {scenario['id']}"
                )
        perceptual_hashes.append((str(scenario["id"]), perceptual_hash))
        for slot in slots:
            polygon = slot.get("polygon", [])
            quality = validate_polygon(polygon)
            if not quality.valid:
                raise PreparationError(f"Invalid polygon in {scenario['id']}: {quality.reason}")
        dataset_counts[str(scenario["dataset"])] += 1
        checked_slots += len(slots)

    missing_datasets = {"PKLot", "CNRPark+EXT", "ACPDS"} - set(dataset_counts)
    if missing_datasets:
        raise PreparationError(f"Catalogue missing datasets: {', '.join(sorted(missing_datasets))}")
    required_minimum = int(catalogue.get("required_minimum_per_dataset", 1))
    under_minimum = {
        name: count for name, count in dataset_counts.items() if count < required_minimum
    }
    if under_minimum:
        raise PreparationError(
            f"Catalogue requires at least {required_minimum} scenarios per dataset: {under_minimum}"
        )
    return {
        "valid": True,
        "catalogue_path": str(catalogue_path),
        "scenario_count": len(scenarios),
        "slot_count": checked_slots,
        "dataset_counts": dict(sorted(dataset_counts.items())),
    }
