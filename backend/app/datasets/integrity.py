from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

import imagehash
from PIL import Image

from app.datasets.preparation import PreparationError

PROTOCOL_ID = "occupancy-v3"
PROTOCOL_SEED = 20260904


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_score(value: str, *, seed: int = PROTOCOL_SEED) -> int:
    return int(hashlib.sha256(f"{seed}|{value}".encode()).hexdigest(), 16)


def perceptual_distance(first: str, second: str) -> int:
    return imagehash.hex_to_hash(first) - imagehash.hex_to_hash(second)


class _PerceptualHashIndex:
    """Exact candidate index for 64-bit hashes within Hamming distance three.

    Splitting a hash into four 16-bit bands guarantees that two values differing in at most three
    bits share at least one unchanged band. This avoids the degenerate traversal cost a BK tree can
    exhibit on a large, visually homogeneous parking-crop corpus.
    """

    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []
        self.bands: dict[tuple[int, int], list[int]] = defaultdict(list)

    @staticmethod
    def _bands(value: str) -> tuple[int, int, int, int]:
        numeric = int(value, 16)
        return tuple((numeric >> (16 * index)) & 0xFFFF for index in range(4))  # type: ignore[return-value]

    def add(self, partition: str, row_id: str, value: str) -> None:
        row_index = len(self.rows)
        self.rows.append((partition, row_id, value))
        for band_index, band in enumerate(self._bands(value)):
            self.bands[(band_index, band)].append(row_index)

    def query(self, value: str, maximum: int) -> list[tuple[str, str]]:
        if maximum > 3:
            raise ValueError("The four-band perceptual index supports distance up to three")
        matches: list[tuple[str, str]] = []
        candidates: set[int] = set()
        for band_index, band in enumerate(self._bands(value)):
            candidates.update(self.bands.get((band_index, band), ()))
        for index in candidates:
            partition, row_id, other = self.rows[index]
            if perceptual_distance(value, other) <= maximum:
                matches.append((partition, row_id))
        return matches


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp"
    ) as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":"), ensure_ascii=False) + "\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def verify_partition_integrity(
    rows: list[dict[str, object]], *, near_duplicate_distance: int = 3
) -> dict[str, object]:
    if not rows:
        raise PreparationError("V2 protocol manifest is empty")
    partitions = {str(row.get("partition")) for row in rows}
    if partitions != {"train", "validation", "holdout"}:
        raise PreparationError(f"Manifest partitions are incomplete: {sorted(partitions)}")

    group_partitions: dict[tuple[str, str], set[str]] = defaultdict(set)
    source_partitions: dict[tuple[str, str], set[str]] = defaultdict(set)
    exact_partitions: dict[str, set[str]] = defaultdict(set)
    perceptual_rows: list[tuple[str, str, str]] = []
    for row in rows:
        dataset = str(row["dataset"])
        partition = str(row["partition"])
        group_partitions[(dataset, str(row["group_id"]))].add(partition)
        source_partitions[(dataset, str(row["source_id"]))].add(partition)
        exact_partitions[str(row["content_sha256"])].add(partition)
        perceptual = str(row.get("perceptual_hash", ""))
        if perceptual:
            perceptual_rows.append((partition, str(row["id"]), perceptual))

    if any(len(value) != 1 for value in group_partitions.values()):
        raise PreparationError("Group leakage crosses V2 partitions")
    if any(len(value) != 1 for value in source_partitions.values()):
        raise PreparationError("Source-image leakage crosses V2 partitions")
    if any(len(value) != 1 for value in exact_partitions.values()):
        raise PreparationError("Exact duplicate content crosses V2 partitions")

    near_duplicates = []
    hash_tree = _PerceptualHashIndex()
    for partition, row_id, value in perceptual_rows:
        for other_partition, other_id in hash_tree.query(value, near_duplicate_distance):
            if partition != other_partition:
                near_duplicates.append([row_id, other_id])
                if len(near_duplicates) >= 25:
                    break
        hash_tree.add(partition, row_id, value)
        if len(near_duplicates) >= 25:
            break
    if near_duplicates:
        raise PreparationError(
            f"Near-duplicate content crosses V2 partitions: {near_duplicates[:3]}"
        )

    return {
        "valid": True,
        "protocol_id": PROTOCOL_ID,
        "rows": len(rows),
        "partitions": {
            partition: sum(str(row["partition"]) == partition for row in rows)
            for partition in ("train", "validation", "holdout")
        },
        "groups": len(group_partitions),
        "sources": len(source_partitions),
        "exact_hashes": len(exact_partitions),
        "near_duplicate_distance": near_duplicate_distance,
    }


def quarantine_cross_partition_duplicates(
    rows: list[dict[str, object]],
    *,
    near_duplicate_distance: int = 3,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Prefer protected evaluation rows and quarantine visually duplicate lower-tier rows."""
    priority = {"holdout": 0, "validation": 1, "train": 2}
    ordered = sorted(
        rows,
        key=lambda row: (
            priority[str(row["partition"])],
            str(row["dataset"]),
            str(row["id"]),
        ),
    )
    kept: list[dict[str, object]] = []
    quarantined: list[dict[str, object]] = []
    exact: dict[str, tuple[str, str]] = {}
    tree = _PerceptualHashIndex()
    for index, row in enumerate(ordered, 1):
        partition = str(row["partition"])
        row_id = str(row["id"])
        digest = str(row["content_sha256"])
        perceptual = str(row.get("perceptual_hash", ""))
        conflict: tuple[str, str] | None = None
        if digest in exact and exact[digest][0] != partition:
            conflict = exact[digest]
        elif perceptual:
            conflict = next(
                (
                    (other_partition, other_id)
                    for other_partition, other_id in tree.query(perceptual, near_duplicate_distance)
                    if other_partition != partition
                ),
                None,
            )
        if conflict:
            quarantined.append(
                {
                    **row,
                    "quarantine_reason": "cross_partition_duplicate_content",
                    "conflicts_with_partition": conflict[0],
                    "conflicts_with_id": conflict[1],
                }
            )
            continue
        kept.append(row)
        exact[digest] = (partition, row_id)
        if perceptual:
            tree.add(partition, row_id, perceptual)
        if progress and index % 10_000 == 0:
            progress(f"V2 duplicate quarantine: {index:,}/{len(ordered):,}")
    kept.sort(key=lambda row: (str(row["dataset"]), str(row["partition"]), str(row["id"])))
    return kept, quarantined


def write_exposure_manifest(
    source_root: Path, artifact_root: Path | None = None
) -> dict[str, object]:
    artifact_root = artifact_root or source_root
    catalogue_paths = [source_root / "demo" / "catalogue.json"]
    artifact_catalogue = artifact_root / "demo" / "catalogue.json"
    if artifact_catalogue not in catalogue_paths:
        catalogue_paths.append(artifact_catalogue)
    catalogue_paths = [path for path in catalogue_paths if path.is_file()]
    if not catalogue_paths:
        raise PreparationError("Prepared catalogue is required before exposure manifest")
    rows = []
    seen_scenarios: set[tuple[str, str]] = set()
    for catalogue_path in catalogue_paths:
        with catalogue_path.open(encoding="utf-8") as handle:
            catalogue = json.load(handle)
        catalogue_root = catalogue_path.parents[1]
        for scenario in catalogue.get("scenarios", []):
            key = (str(scenario["dataset"]), str(scenario["id"]))
            if key in seen_scenarios:
                continue
            seen_scenarios.add(key)
            image_path = catalogue_root / str(scenario["image_path"])
            source_sha256 = scenario.get("source_sha256")
            perceptual_hash = scenario.get("perceptual_hash")
            if image_path.is_file() and not source_sha256:
                source_sha256 = sha256_file(image_path)
            if image_path.is_file() and not perceptual_hash:
                with Image.open(image_path) as image:
                    perceptual_hash = str(imagehash.phash(image.convert("RGB")))
            rows.append(
                {
                    "id": str(scenario["id"]),
                    "dataset": str(scenario["dataset"]),
                    "lot": str(scenario.get("lot", "")),
                    "condition": str(scenario.get("condition", "")),
                    "split": str(scenario.get("split", "")),
                    "source_sha256": str(source_sha256 or ""),
                    "perceptual_hash": str(perceptual_hash or ""),
                    "exposure": "prepared_scenario",
                }
            )
    benchmark_manifest = source_root / "prepared" / "benchmark" / "manifest.jsonl"
    if benchmark_manifest.is_file():
        for row in read_jsonl(benchmark_manifest):
            rows.append(
                {
                    "id": str(row.get("id")),
                    "dataset": str(row.get("dataset")),
                    "source_id": str(row.get("source_id")),
                    "content_sha256": str(row.get("content_sha256")),
                    "exposure": f"v1_{row.get('partition', 'benchmark')}",
                }
            )
    rows.sort(key=lambda row: (str(row["exposure"]), str(row["dataset"]), str(row["id"])))
    root = artifact_root / "prepared" / "v2-protocol"
    manifest_path = root / "historical-exposure.jsonl"
    atomic_jsonl(manifest_path, rows)
    report = {
        "protocol_id": PROTOCOL_ID,
        "generated_at": datetime.now(UTC).isoformat(),
        "historically_virgin": False,
        "protection_statement": "Protected from the integrated refinement point forward",
        "row_count": len(rows),
        "catalogues": [str(path) for path in catalogue_paths],
        "manifest": manifest_path.relative_to(artifact_root).as_posix(),
        "manifest_sha256": sha256_file(manifest_path),
        "external_confirmation": {
            "required_for_historically_independent_claim": True,
            "status": "not_provided",
            "ingestion_contract": "JSONL source manifest plus authorized images and slot polygons",
        },
    }
    atomic_json(root / "exposure-report.json", report)
    return report
