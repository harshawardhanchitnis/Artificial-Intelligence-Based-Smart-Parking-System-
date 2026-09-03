from __future__ import annotations

import hashlib
import heapq
import io
import json
import tarfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from PIL import Image

from app.datasets.preparation import (
    ACPDS_ARCHIVE_CANDIDATES,
    CNR_EXT_PATCHES,
    CNR_SPLITS,
    PKLOT_ARCHIVE,
    PreparationError,
    _acpds_archive,
    _atomic_json,
    _atomic_jsonl,
)
from app.ml.features import crop_slot

BENCHMARK_SCHEMA_VERSION = "1.0"
PARTITIONS = ("train", "validation", "test")
DEFAULT_LIMITS = {"train": 1000, "validation": 300, "test": 300}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _score(seed: int, value: str) -> int:
    return int(hashlib.sha256(f"{seed}|{value}".encode()).hexdigest(), 16)


def _pklot_partition(seed: int, group_id: str) -> str:
    bucket = _score(seed, f"PKLot|{group_id}") % 100
    if bucket < 70:
        return "train"
    if bucket < 85:
        return "validation"
    return "test"


def _offer(
    heaps: dict[tuple[str, int], list[tuple[int, str, dict[str, object]]]],
    row: dict[str, object],
    *,
    seed: int,
    limits: dict[str, int],
) -> None:
    partition = str(row["partition"])
    label = int(row["label"])
    limit = limits[partition]
    sample_id = str(row["id"])
    item = (-_score(seed, sample_id), sample_id, row)
    heap = heaps[(partition, label)]
    if len(heap) < limit:
        heapq.heappush(heap, item)
    elif item[0] > heap[0][0]:
        heapq.heapreplace(heap, item)


def _flatten(
    heaps: dict[tuple[str, int], list[tuple[int, str, dict[str, object]]]],
) -> list[dict[str, object]]:
    return sorted(
        [item[2] for heap in heaps.values() for item in heap],
        key=lambda row: (str(row["dataset"]), str(row["partition"]), str(row["id"])),
    )


def _write_patch(
    root: Path,
    row: dict[str, object],
    source: BinaryIO | io.BytesIO,
    *,
    force: bool,
) -> dict[str, object]:
    relative = Path("patches") / str(row["dataset_slug"]) / str(row["partition"])
    relative /= f"{hashlib.sha256(str(row['id']).encode()).hexdigest()[:20]}.jpg"
    destination = root / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = source.read()
    if force or not destination.is_file():
        destination.write_bytes(payload)
    return {key: value for key, value in row.items() if key != "dataset_slug"} | {
        "patch_path": relative.as_posix(),
        "content_sha256": _sha256_bytes(payload),
    }


def _prepare_pklot(
    data_root: Path,
    output_root: Path,
    *,
    seed: int,
    limits: dict[str, int],
    force: bool,
    progress,
) -> list[dict[str, object]]:
    archive_path = data_root / PKLOT_ARCHIVE
    heaps: dict[tuple[str, int], list[tuple[int, str, dict[str, object]]]] = defaultdict(list)
    if progress:
        progress("Benchmark PKLot: selecting segmented patches by site/date groups")
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for index, member in enumerate(archive, start=1):
            parts = PurePosixPath(member.name).parts
            if (
                not member.isfile()
                or len(parts) != 7
                or parts[:2] != ("PKLot", "PKLotSegmented")
                or Path(parts[-1]).suffix.lower() != ".jpg"
                or parts[-2] not in {"Empty", "Occupied"}
            ):
                continue
            _, _, site, weather, date, label_name, filename = parts
            source_id = f"{site}/{date}/{Path(filename).stem.split('#')[0]}"
            group_id = f"{site}/{date}"
            row = {
                "id": f"PKLot/{site}/{weather}/{date}/{filename}",
                "dataset": "PKLot",
                "dataset_slug": "pklot",
                "partition": _pklot_partition(seed, group_id),
                "group_id": group_id,
                "source_id": source_id,
                "label": int(label_name == "Occupied"),
                "source_archive": PKLOT_ARCHIVE.as_posix(),
                "source_member": member.name,
            }
            _offer(heaps, row, seed=seed, limits=limits)
            if progress and index % 250_000 == 0:
                progress(f"Benchmark PKLot: scanned {index:,} archive members")
    selected = _flatten(heaps)
    members = {str(row["source_member"]): row for row in selected}
    written: list[dict[str, object]] = []
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            row = members.get(member.name)
            if row is None:
                continue
            source = archive.extractfile(member)
            if source is not None:
                written.append(_write_patch(output_root, row, source, force=force))
            if len(written) == len(selected):
                break
    if progress:
        progress(f"Benchmark PKLot: prepared {len(written):,} labelled patches")
    return written


def _cnr_labels(data_root: Path, seed: int) -> tuple[list[dict[str, object]], list[str]]:
    with zipfile.ZipFile(data_root / CNR_SPLITS) as splits:
        train_lines = splits.read("splits/CNRPark-EXT/train.txt").decode().splitlines()
        test_lines = splits.read("splits/CNRPark-EXT/test.txt").decode().splitlines()
    train_days = sorted({line.split("/", 2)[1] for line in train_lines if line})
    validation_days = sorted(train_days, key=lambda day: _score(seed, f"CNR|{day}"))[:3]
    validation_set = set(validation_days)
    rows: list[dict[str, object]] = []
    for source_partition, lines in (("train", train_lines), ("test", test_lines)):
        for line in lines:
            if not line:
                continue
            relative, label = line.rsplit(" ", 1)
            parts = PurePosixPath(relative).parts
            date = parts[1]
            partition = (
                "test"
                if source_partition == "test"
                else "validation"
                if date in validation_set
                else "train"
            )
            stem = Path(parts[-1]).stem
            source_id = "/".join((*parts[:-1], stem.rsplit("_", 1)[0]))
            rows.append(
                {
                    "id": f"CNRPark+EXT/{relative}",
                    "dataset": "CNRPark+EXT",
                    "dataset_slug": "cnrpark-ext",
                    "partition": partition,
                    "group_id": date,
                    "source_id": source_id,
                    "label": int(label),
                    "source_archive": CNR_EXT_PATCHES.as_posix(),
                    "source_member": f"PATCHES/{relative}",
                }
            )
    return rows, validation_days


def _prepare_cnr(
    data_root: Path,
    output_root: Path,
    *,
    seed: int,
    limits: dict[str, int],
    force: bool,
    progress,
) -> tuple[list[dict[str, object]], list[str]]:
    candidates, validation_days = _cnr_labels(data_root, seed)
    heaps: dict[tuple[str, int], list[tuple[int, str, dict[str, object]]]] = defaultdict(list)
    for row in candidates:
        _offer(heaps, row, seed=seed, limits=limits)
    selected = _flatten(heaps)
    with zipfile.ZipFile(data_root / CNR_EXT_PATCHES) as archive:
        rows = [
            _write_patch(output_root, row, archive.open(str(row["source_member"])), force=force)
            for row in selected
        ]
    if progress:
        progress(f"Benchmark CNRPark+EXT: prepared {len(rows):,} labelled patches")
    return rows, validation_days


def _prepare_acpds(
    data_root: Path,
    output_root: Path,
    *,
    seed: int,
    limits: dict[str, int],
    force: bool,
    progress,
) -> list[dict[str, object]]:
    archive_path = _acpds_archive(data_root)
    heaps: dict[tuple[str, int], list[tuple[int, str, dict[str, object]]]] = defaultdict(list)
    with zipfile.ZipFile(archive_path) as archive:
        annotations = json.loads(archive.read("annotations.json"))
        for source_split, partition in (
            ("train", "train"),
            ("valid", "validation"),
            ("test", "test"),
        ):
            split = annotations[source_split]
            for image_index, filename in enumerate(split["file_names"]):
                source_id = Path(str(filename)).stem
                for slot_index, label in enumerate(split["occupancy_list"][image_index]):
                    _offer(
                        heaps,
                        {
                            "id": f"ACPDS/{source_split}/{source_id}/{slot_index + 1}",
                            "dataset": "ACPDS",
                            "dataset_slug": "acpds",
                            "partition": partition,
                            "group_id": source_id,
                            "source_id": source_id,
                            "label": int(bool(label)),
                            "source_archive": archive_path.relative_to(data_root).as_posix(),
                            "source_member": f"images/{filename}",
                            "image_index": image_index,
                            "slot_index": slot_index,
                            "source_split": source_split,
                        },
                        seed=seed,
                        limits=limits,
                    )
        selected = _flatten(heaps)
        image_cache: dict[str, Image.Image] = {}
        rows: list[dict[str, object]] = []
        for row in selected:
            member = str(row["source_member"])
            image = image_cache.get(member)
            if image is None:
                image = Image.open(io.BytesIO(archive.read(member))).convert("RGB")
                image_cache[member] = image
            split = annotations[str(row["source_split"])]
            polygon = split["rois_list"][int(row["image_index"])][int(row["slot_index"])]
            normalized = [[float(x), float(y)] for x, y in polygon]
            patch = crop_slot(image, normalized)
            buffer = io.BytesIO()
            patch.save(buffer, format="JPEG", quality=95)
            buffer.seek(0)
            clean = {
                key: value
                for key, value in row.items()
                if key not in {"image_index", "slot_index", "source_split"}
            }
            rows.append(_write_patch(output_root, clean, buffer, force=force))
    if progress:
        progress(f"Benchmark ACPDS: prepared {len(rows):,} labelled patches")
    return rows


def _counts(rows: list[dict[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for partition in PARTITIONS:
        selected = [row for row in rows if row["partition"] == partition]
        result[partition] = {
            "samples": len(selected),
            "vacant": sum(int(row["label"]) == 0 for row in selected),
            "occupied": sum(int(row["label"]) == 1 for row in selected),
            "groups": len({(row["dataset"], row["group_id"]) for row in selected}),
            "sources": len({(row["dataset"], row["source_id"]) for row in selected}),
            "datasets": dict(sorted(Counter(str(row["dataset"]) for row in selected).items())),
        }
    return result


def _remove_cross_partition_duplicates(
    rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], int]:
    seen: dict[str, str] = {}
    kept: list[dict[str, object]] = []
    removed = 0
    for partition in PARTITIONS:
        for row in (item for item in rows if item["partition"] == partition):
            digest = str(row["content_sha256"])
            prior = seen.get(digest)
            if prior is not None and prior != partition:
                removed += 1
                continue
            seen[digest] = partition
            kept.append(row)
    return kept, removed


def prepare_benchmark(
    data_root: Path,
    *,
    samples_per_class: dict[str, int] | None = None,
    random_seed: int = 42,
    force: bool = False,
    progress=print,
) -> dict[str, object]:
    data_root = data_root.resolve()
    limits = dict(samples_per_class or DEFAULT_LIMITS)
    if set(limits) != set(PARTITIONS) or any(value < 50 for value in limits.values()):
        raise PreparationError(
            "Benchmark limits must define train/validation/test with at least 50 samples per class"
        )
    output_root = data_root / "prepared" / "ml-benchmark"
    rows = _prepare_pklot(
        data_root, output_root, seed=random_seed, limits=limits, force=force, progress=progress
    )
    cnr_rows, cnr_validation_days = _prepare_cnr(
        data_root, output_root, seed=random_seed, limits=limits, force=force, progress=progress
    )
    rows.extend(cnr_rows)
    rows.extend(
        _prepare_acpds(
            data_root, output_root, seed=random_seed, limits=limits, force=force, progress=progress
        )
    )
    rows, duplicates_removed = _remove_cross_partition_duplicates(rows)
    manifest_path = output_root / "manifest.jsonl"
    _atomic_jsonl(manifest_path, rows)
    report = {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "profile": "ml-benchmark",
        "random_seed": random_seed,
        "samples_per_class_limits": limits,
        "manifest_path": manifest_path.relative_to(data_root).as_posix(),
        "manifest_sha256": _sha256_file(manifest_path),
        "duplicate_content_samples_removed": duplicates_removed,
        "partition_counts": _counts(rows),
        "split_definitions": {
            "PKLot": "site/date groups assigned 70/15/15 by seeded SHA-256; segmented patch labels",
            "CNRPark+EXT": (
                "official test days retained; three seeded train days reserved for validation"
            ),
            "CNRPark+EXT_validation_days": cnr_validation_days,
            "ACPDS": "official train/valid/test image splits retained",
        },
        "source_archives": [
            PKLOT_ARCHIVE.as_posix(),
            CNR_EXT_PATCHES.as_posix(),
            CNR_SPLITS.as_posix(),
            next(
                path for path in ACPDS_ARCHIVE_CANDIDATES if (data_root / path).is_file()
            ).as_posix(),
        ],
    }
    _atomic_json(output_root / "report.json", report)
    return report


def verify_benchmark(data_root: Path) -> dict[str, object]:
    root = data_root.resolve() / "prepared" / "ml-benchmark"
    manifest_path = root / "manifest.jsonl"
    report_path = root / "report.json"
    if not manifest_path.is_file() or not report_path.is_file():
        raise PreparationError("ML benchmark is missing; run the Benchmark preparation profile")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise PreparationError("Unsupported ML benchmark schema")
    if report.get("manifest_sha256") != _sha256_file(manifest_path):
        raise PreparationError("ML benchmark manifest checksum mismatch")
    rows = [
        json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line
    ]
    group_partitions: dict[tuple[str, str], set[str]] = defaultdict(set)
    source_partitions: dict[tuple[str, str], set[str]] = defaultdict(set)
    content_partitions: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        partition = str(row.get("partition"))
        if partition not in PARTITIONS:
            raise PreparationError(f"Invalid benchmark partition: {partition}")
        patch_path = (root / str(row["patch_path"])).resolve()
        if not patch_path.is_relative_to(root) or not patch_path.is_file():
            raise PreparationError(f"Benchmark patch is missing or unsafe: {row.get('id')}")
        if row.get("content_sha256") != _sha256_file(patch_path):
            raise PreparationError(f"Benchmark patch checksum mismatch: {row.get('id')}")
        group_partitions[(str(row["dataset"]), str(row["group_id"]))].add(partition)
        source_partitions[(str(row["dataset"]), str(row["source_id"]))].add(partition)
        content_partitions[str(row["content_sha256"])].add(partition)
    if any(len(value) > 1 for value in group_partitions.values()):
        raise PreparationError("Benchmark group overlap detected")
    if any(len(value) > 1 for value in source_partitions.values()):
        raise PreparationError("Benchmark source overlap detected")
    if any(len(value) > 1 for value in content_partitions.values()):
        raise PreparationError("Duplicate patch content crosses benchmark partitions")
    counts = _counts(rows)
    for partition in PARTITIONS:
        section = counts[partition]
        if not section["vacant"] or not section["occupied"]:
            raise PreparationError(f"Benchmark {partition} partition lacks both classes")
    return {
        "valid": True,
        "manifest_sha256": report["manifest_sha256"],
        "sample_count": len(rows),
        "partition_counts": counts,
        "split_integrity": True,
    }
