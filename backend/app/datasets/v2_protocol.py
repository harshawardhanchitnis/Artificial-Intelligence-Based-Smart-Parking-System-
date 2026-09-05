from __future__ import annotations

import csv
import hashlib
import heapq
import io
import json
import re
import tarfile
import zipfile
from collections import defaultdict
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from xml.etree import ElementTree

import imagehash
from PIL import Image, ImageOps

from app.datasets.integrity import (
    PROTOCOL_ID,
    PROTOCOL_SEED,
    atomic_json,
    atomic_jsonl,
    deterministic_score,
    quarantine_cross_partition_duplicates,
    read_jsonl,
    sha256_file,
    verify_partition_integrity,
    write_exposure_manifest,
)
from app.datasets.preparation import (
    ACPDS_ARCHIVE_CANDIDATES,
    CNR_FULL_ARCHIVE,
    CNR_LABELS,
    CNR_SPLITS,
    PKLOT_ARCHIVE,
    PreparationError,
)
from app.ml.geometry import order_polygon, rectify_slot, validate_polygon

ProgressCallback = Callable[[str], None]
PARTITIONS = ("train", "validation", "holdout")
STANDARD_SOURCE_LIMITS = {
    "PKLot": {"train": 1_250, "validation": 350, "holdout": 400},
    "CNRPark+EXT": {"train": 1_250, "validation": 300, "holdout": 400},
    "ACPDS": {"train": 231, "validation": 35, "holdout": 27},
}
SMOKE_SOURCE_LIMITS = {
    dataset: {partition: 4 for partition in PARTITIONS}
    for dataset in ("PKLot", "CNRPark+EXT", "ACPDS")
}
STANDARD_CROP_LIMITS = {
    "PKLot": {"train": 48_000, "validation": 8_000, "holdout": 12_000},
    "CNRPark+EXT": {"train": 48_000, "validation": 8_000, "holdout": 12_000},
    "ACPDS": {"train": 7_842, "validation": 1_904, "holdout": 1_490},
}
SMOKE_CROP_LIMITS = {
    dataset: {"train": 120, "validation": 40, "holdout": 40}
    for dataset in ("PKLot", "CNRPark+EXT", "ACPDS")
}


def _emit(progress: ProgressCallback | None, message: str) -> None:
    if progress:
        progress(message)


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")


def _official_group_partitions(source_root: Path, dataset: str) -> dict[tuple[str, str], str]:
    mapping: dict[tuple[str, str], str] = {}
    names = {
        "train": f"splits/{dataset}/train.txt",
        "validation": f"splits/{dataset}/val.txt",
        "holdout": f"splits/{dataset}/test.txt",
    }
    with zipfile.ZipFile(source_root / CNR_SPLITS) as archive:
        for partition, name in names.items():
            for line in archive.read(name).decode(errors="replace").splitlines():
                parts = PurePosixPath(line.strip()).parts
                date = next(
                    (part for part in parts if re.fullmatch(r"\d{4}-\d{2}-\d{2}", part)), None
                )
                if not date:
                    continue
                if dataset == "PKLot":
                    site = next(
                        (
                            part
                            for part in parts
                            if part.upper() in {"PUC", "PUCPR", "UFPR04", "UFPR05"}
                        ),
                        None,
                    )
                    if site:
                        mapping[("PUCPR" if site.upper() == "PUC" else site.upper(), date)] = (
                            partition
                        )
                else:
                    mapping[("day", date)] = partition
    return mapping


def _reservoir_add(
    heaps: dict[str, list[tuple[int, str]]], partition: str, source_id: str, limit: int
) -> None:
    score = deterministic_score(source_id)
    heap = heaps[partition]
    value = (-score, source_id)
    if len(heap) < limit:
        heapq.heappush(heap, value)
    elif value > heap[0]:
        heapq.heapreplace(heap, value)


def _selected(heaps: dict[str, list[tuple[int, str]]]) -> set[str]:
    return {source_id for heap in heaps.values() for _, source_id in heap}


def _save_source(
    destination: Path,
    payload: bytes,
    *,
    dataset: str,
    partition: str,
    source_id: str,
    group_id: str,
    condition: str,
    slots: list[dict[str, object]],
    localization_partition: str,
) -> dict[str, object]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    with Image.open(io.BytesIO(payload)) as image:
        rgb = ImageOps.exif_transpose(image).convert("RGB")
        perceptual = str(imagehash.phash(rgb))
        width, height = rgb.size
    return {
        "dataset": dataset,
        "partition": partition,
        "localization_partition": localization_partition,
        "source_id": source_id,
        "group_id": group_id,
        "condition": condition,
        "image_path": destination.as_posix(),
        "image_width": width,
        "image_height": height,
        "source_sha256": sha256_file(destination),
        "perceptual_hash": perceptual,
        "slots": slots,
    }


def _pklot_slots(xml_payload: bytes, width: int, height: int) -> list[dict[str, object]]:
    root = ElementTree.fromstring(xml_payload)
    slots = []
    for space in root.findall("space"):
        contour = space.find("contour")
        if contour is None:
            continue
        polygon = [
            [float(point.attrib["x"]) / width, float(point.attrib["y"]) / height]
            for point in contour.findall("point")
        ]
        polygon = order_polygon(polygon)
        if validate_polygon(polygon).valid:
            slots.append(
                {
                    "id": str(space.attrib.get("id", len(slots) + 1)),
                    "polygon": polygon,
                    "occupied": space.attrib.get("occupied") == "1",
                }
            )
    return slots


def _prepare_pklot_sources(
    source_root: Path,
    output_root: Path,
    limits: dict[str, int],
    progress: ProgressCallback | None,
) -> list[dict[str, object]]:
    groups = _official_group_partitions(source_root, "PKLot")
    heaps = {partition: [] for partition in PARTITIONS}
    archive_path = source_root / PKLOT_ARCHIVE
    _emit(progress, "V2 PKLot: indexing full-image sources")
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) != 6 or parts[:2] != ("PKLot", "PKLot"):
                continue
            if Path(parts[-1]).suffix.lower() != ".jpg":
                continue
            site, weather, date, filename = parts[2], parts[3], parts[4], parts[5]
            partition = groups.get((site.upper(), date))
            if partition:
                _reservoir_add(
                    heaps,
                    partition,
                    f"{site}/{weather}/{date}/{Path(filename).stem}",
                    limits[partition],
                )
    selected = _selected(heaps)
    records: dict[str, dict[str, object]] = {source_id: {} for source_id in selected}
    _emit(progress, f"V2 PKLot: extracting {len(selected):,} selected sources")
    with tarfile.open(archive_path, "r|gz") as archive:
        for member in archive:
            parts = PurePosixPath(member.name).parts
            if not member.isfile() or len(parts) != 6 or parts[:2] != ("PKLot", "PKLot"):
                continue
            suffix = Path(parts[-1]).suffix.lower()
            if suffix not in {".jpg", ".xml"}:
                continue
            site, weather, date, filename = parts[2], parts[3], parts[4], parts[5]
            source_id = f"{site}/{weather}/{date}/{Path(filename).stem}"
            if source_id not in records:
                continue
            handle = archive.extractfile(member)
            if handle:
                records[source_id][suffix] = handle.read()

    sources = []
    site_localization = {"PUCPR": "train", "UFPR04": "validation", "UFPR05": "holdout"}
    for source_id, payloads in sorted(records.items()):
        if ".jpg" not in payloads or ".xml" not in payloads:
            continue
        parts = source_id.split("/")
        site, weather, date = parts[:3]
        partition = groups[(site.upper(), date)]
        image_payload = payloads[".jpg"]
        with Image.open(io.BytesIO(image_payload)) as image:
            width, height = image.size
        slots = _pklot_slots(payloads[".xml"], width, height)
        if not slots:
            continue
        destination = output_root / "sources" / "pklot" / f"{_slug(source_id)}.jpg"
        sources.append(
            _save_source(
                destination,
                image_payload,
                dataset="PKLot",
                partition=partition,
                source_id=source_id,
                group_id=f"{site}/{date}",
                condition=weather,
                slots=slots,
                localization_partition=site_localization.get(site.upper(), "train"),
            )
        )
    return sources


def _cnr_datetime(value: str) -> str:
    date, time = value.split("_", 1)
    return f"{date}_{time.replace('.', '')}"


def _prepare_cnr_sources(
    source_root: Path,
    output_root: Path,
    limits: dict[str, int],
    progress: ProgressCallback | None,
) -> list[dict[str, object]]:
    day_partitions = _official_group_partitions(source_root, "CNRPark-EXT")
    heaps = {partition: [] for partition in PARTITIONS}
    archive_path = source_root / CNR_FULL_ARCHIVE
    member_meta: dict[str, tuple[str, str, str, str]] = {}
    geometry_payloads: dict[str, bytes] = {}
    _emit(progress, "V2 CNRPark+EXT: indexing full-image sources")
    with tarfile.open(archive_path, "r|") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if (
                Path(member.name).name.lower().startswith("camera")
                and Path(member.name).suffix == ".csv"
            ):
                handle = archive.extractfile(member)
                if handle:
                    geometry_payloads[Path(member.name).stem.lower()] = handle.read()
                continue
            if len(parts) != 5 or parts[0] != "FULL_IMAGE_1000x750":
                continue
            weather, date, camera, filename = parts[1], parts[2], parts[3].lower(), parts[4]
            partition = day_partitions.get(("day", date))
            if not partition:
                continue
            source_id = f"{camera}/{weather}/{date}/{Path(filename).stem}"
            _reservoir_add(heaps, partition, source_id, limits[partition])
            member_meta[source_id] = (member.name, camera, weather, date)
    selected = _selected(heaps)
    image_payloads: dict[str, bytes] = {}
    with tarfile.open(archive_path, "r|") as archive:
        for member in archive:
            if not member.isfile():
                continue
            parts = PurePosixPath(member.name).parts
            if len(parts) != 5 or parts[0] != "FULL_IMAGE_1000x750":
                continue
            source_id = f"{parts[3].lower()}/{parts[1]}/{parts[2]}/{Path(parts[4]).stem}"
            if source_id in selected:
                handle = archive.extractfile(member)
                if handle:
                    image_payloads[source_id] = handle.read()

    geometry: dict[str, dict[str, tuple[float, float, float, float]]] = defaultdict(dict)
    for camera, payload in geometry_payloads.items():
        for row in csv.DictReader(io.StringIO(payload.decode("utf-8-sig"))):
            geometry[camera][str(row["SlotId"])] = tuple(
                float(row[key]) for key in ("X", "Y", "W", "H")
            )
    occupancy: dict[tuple[str, str], list[tuple[str, bool]]] = defaultdict(list)
    with (source_root / CNR_LABELS).open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if not str(row.get("camera", "")).isdigit():
                continue
            camera = f"camera{int(row['camera'])}"
            stem = _cnr_datetime(str(row["datetime"]))
            occupancy[(camera, stem)].append((str(row["slot_id"]), str(row["occupancy"]) == "1"))

    sources = []
    for source_id, image_payload in sorted(image_payloads.items()):
        _, camera, weather, date = member_meta[source_id]
        stem = source_id.rsplit("/", 1)[-1]
        slots = []
        for slot_id, occupied in occupancy.get((camera, stem), []):
            rectangle = geometry[camera].get(slot_id)
            if not rectangle:
                continue
            x, y, width, height = rectangle
            polygon = order_polygon(
                [
                    [x / 2592.0, y / 1944.0],
                    [(x + width) / 2592.0, y / 1944.0],
                    [(x + width) / 2592.0, (y + height) / 1944.0],
                    [x / 2592.0, (y + height) / 1944.0],
                ]
            )
            if validate_polygon(polygon).valid:
                slots.append({"id": slot_id, "polygon": polygon, "occupied": occupied})
        if not slots:
            continue
        partition = day_partitions[("day", date)]
        camera_number = int(camera.removeprefix("camera"))
        localization_partition = (
            "train" if camera_number <= 6 else "validation" if camera_number == 7 else "holdout"
        )
        destination = output_root / "sources" / "cnrpark-ext" / f"{_slug(source_id)}.jpg"
        sources.append(
            _save_source(
                destination,
                image_payload,
                dataset="CNRPark+EXT",
                partition=partition,
                source_id=source_id,
                group_id=f"{camera}/{date}",
                condition=weather,
                slots=slots,
                localization_partition=localization_partition,
            )
        )
    return sources


def _acpds_archive(source_root: Path) -> Path:
    for relative in ACPDS_ARCHIVE_CANDIDATES:
        path = source_root / relative
        if path.is_file():
            return path
    raise PreparationError("ACPDS archive is missing")


def _prepare_acpds_sources(
    source_root: Path,
    output_root: Path,
    limits: dict[str, int],
    progress: ProgressCallback | None,
) -> list[dict[str, object]]:
    split_mapping = {"train": "train", "valid": "validation", "test": "holdout"}
    sources = []
    with zipfile.ZipFile(_acpds_archive(source_root)) as archive:
        annotations = json.loads(archive.read("annotations.json"))
        for source_split, partition in split_mapping.items():
            split = annotations[source_split]
            indices = sorted(
                range(len(split["file_names"])),
                key=lambda index: deterministic_score(
                    f"ACPDS/{source_split}/{split['file_names'][index]}"
                ),
            )[: limits[partition]]
            for index in indices:
                filename = str(split["file_names"][index])
                payload = archive.read(f"images/{filename}")
                slots = []
                for slot_index, (polygon, occupied) in enumerate(
                    zip(split["rois_list"][index], split["occupancy_list"][index], strict=True),
                    start=1,
                ):
                    ordered = order_polygon([[float(x), float(y)] for x, y in polygon])
                    if validate_polygon(ordered).valid:
                        slots.append(
                            {"id": str(slot_index), "polygon": ordered, "occupied": bool(occupied)}
                        )
                source_id = f"{source_split}/{Path(filename).stem}"
                destination = output_root / "sources" / "acpds" / f"{_slug(source_id)}.jpg"
                sources.append(
                    _save_source(
                        destination,
                        payload,
                        dataset="ACPDS",
                        partition=partition,
                        source_id=source_id,
                        group_id=source_id,
                        condition="mixed",
                        slots=slots,
                        localization_partition=partition,
                    )
                )
    _emit(progress, f"V2 ACPDS: prepared {len(sources):,} full-image sources")
    return sources


def _write_crops(
    sources: list[dict[str, object]],
    output_root: Path,
    limits: dict[str, dict[str, int]],
    progress: ProgressCallback | None,
) -> list[dict[str, object]]:
    candidates: dict[tuple[str, str, bool], list[tuple[int, dict[str, object]]]] = defaultdict(list)
    for source in sources:
        dataset = str(source["dataset"])
        partition = str(source["partition"])
        for slot in source["slots"]:
            row_id = f"{dataset}/{source['source_id']}/{slot['id']}"
            candidates[(dataset, partition, bool(slot["occupied"]))].append(
                (deterministic_score(row_id), {"id": row_id, "source": source, "slot": slot})
            )

    selected: list[dict[str, object]] = []
    for dataset in ("PKLot", "CNRPark+EXT", "ACPDS"):
        for partition in PARTITIONS:
            maximum = limits[dataset][partition]
            per_class = maximum // 2
            for occupied in (False, True):
                rows = sorted(candidates[(dataset, partition, occupied)], key=lambda item: item[0])
                selected.extend(row for _, row in rows[:per_class])

    output_rows = []
    for index, selected_row in enumerate(selected, start=1):
        source = selected_row["source"]
        slot = selected_row["slot"]
        relative = (
            Path("crops")
            / str(source["partition"])
            / _slug(str(source["dataset"]))
            / f"{hashlib_sha(str(selected_row['id']))}.jpg"
        )
        destination = output_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file():
            try:
                with Image.open(destination) as existing:
                    patch = existing.convert("RGB").copy()
            except OSError:
                destination.unlink(missing_ok=True)
                with Image.open(str(source["image_path"])) as image:
                    patch = rectify_slot(ImageOps.exif_transpose(image), slot["polygon"])
                patch.save(destination, format="JPEG", quality=92, optimize=True)
        else:
            with Image.open(str(source["image_path"])) as image:
                patch = rectify_slot(ImageOps.exif_transpose(image), slot["polygon"])
            patch.save(destination, format="JPEG", quality=92, optimize=True)
        output_rows.append(
            {
                "id": selected_row["id"],
                "dataset": source["dataset"],
                "partition": source["partition"],
                "group_id": source["group_id"],
                "source_id": source["source_id"],
                "condition": source["condition"],
                "label": int(bool(slot["occupied"])),
                "patch_path": relative.as_posix(),
                "content_sha256": sha256_file(destination),
                "perceptual_hash": str(imagehash.phash(patch)),
                "preprocessing": "perspective-mask-v2-128",
            }
        )
        if index % 2_000 == 0:
            _emit(progress, f"V2 crops: {index:,}/{len(selected):,}")
    return output_rows


def hashlib_sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def prepare_v2_protocol(
    source_root: Path,
    artifact_root: Path | None = None,
    *,
    profile: str = "standard",
    progress: ProgressCallback | None = print,
) -> dict[str, object]:
    if profile not in {"smoke", "standard"}:
        raise PreparationError("V2 profile must be smoke or standard")
    source_root = source_root.resolve()
    artifact_root = (artifact_root or source_root).resolve()
    output_root = artifact_root / "prepared" / "v2-protocol"
    source_limits = SMOKE_SOURCE_LIMITS if profile == "smoke" else STANDARD_SOURCE_LIMITS
    crop_limits = SMOKE_CROP_LIMITS if profile == "smoke" else STANDARD_CROP_LIMITS
    exposure = write_exposure_manifest(source_root, artifact_root)
    sources = []
    sources.extend(
        _prepare_pklot_sources(source_root, output_root, source_limits["PKLot"], progress)
    )
    sources.extend(
        _prepare_cnr_sources(source_root, output_root, source_limits["CNRPark+EXT"], progress)
    )
    sources.extend(
        _prepare_acpds_sources(source_root, output_root, source_limits["ACPDS"], progress)
    )
    exposure_rows = read_jsonl(output_root / "historical-exposure.jsonl")
    exposed_source_hashes = {
        str(row.get("source_sha256"))
        for row in exposure_rows
        if row.get("exposure") == "prepared_scenario" and row.get("source_sha256")
    }
    exposed_groups: set[tuple[str, str]] = set()
    for row in exposure_rows:
        if row.get("exposure") != "prepared_scenario":
            continue
        dataset = str(row.get("dataset"))
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", str(row.get("id", "")))
        lot = str(row.get("lot", ""))
        if dataset in {"PKLot", "CNRPark+EXT"} and lot and date_match:
            exposed_groups.add((dataset, f"{lot}/{date_match.group()}"))
    exposed_sources = [
        source for source in sources if str(source["source_sha256"]) in exposed_source_hashes
    ]
    exposed_groups.update(
        (str(source["dataset"]), str(source["group_id"])) for source in exposed_sources
    )
    sources = [
        source
        for source in sources
        if str(source["source_sha256"]) not in exposed_source_hashes
        and (str(source["dataset"]), str(source["group_id"])) not in exposed_groups
    ]
    _emit(
        progress,
        f"V2 exposure guard: excluded {len(exposed_sources):,} prepared-scenario sources",
    )
    source_manifest = output_root / "source-manifest.jsonl"
    serializable_sources = [
        {
            **source,
            "image_path": Path(str(source["image_path"])).relative_to(output_root).as_posix(),
        }
        for source in sources
    ]
    atomic_jsonl(source_manifest, serializable_sources)
    rows = _write_crops(sources, output_root, crop_limits, progress)
    rows, quarantined = quarantine_cross_partition_duplicates(rows, progress=progress)
    manifest = output_root / "occupancy-manifest.jsonl"
    atomic_jsonl(manifest, rows)
    quarantine_manifest = output_root / "quarantined-duplicates.jsonl"
    atomic_jsonl(quarantine_manifest, quarantined)
    integrity = verify_partition_integrity(rows)
    report = {
        "protocol_id": PROTOCOL_ID,
        "seed": PROTOCOL_SEED,
        "profile": profile,
        "generated_at": datetime.now(UTC).isoformat(),
        "protected_from": "integrated refinement point forward",
        "historically_virgin": False,
        "preprocessing": "perspective-mask-v2-128",
        "sources": len(sources),
        "excluded_exposed_sources": len(exposed_sources),
        "excluded_exposed_groups": len(exposed_groups),
        "samples": len(rows),
        "quarantined_cross_partition_duplicates": len(quarantined),
        "partitions": integrity["partitions"],
        "datasets": {
            dataset: sum(str(row["dataset"]) == dataset for row in rows)
            for dataset in ("PKLot", "CNRPark+EXT", "ACPDS")
        },
        "source_manifest_sha256": sha256_file(source_manifest),
        "occupancy_manifest_sha256": sha256_file(manifest),
        "quarantine_manifest_sha256": sha256_file(quarantine_manifest),
        "integrity": integrity,
        "historical_exposure_manifest_sha256": exposure["manifest_sha256"],
        "final_holdout_policy": (
            "single use after architecture, weights, threshold and calibration freeze"
        ),
    }
    atomic_json(output_root / "protocol-report.json", report)
    return report
