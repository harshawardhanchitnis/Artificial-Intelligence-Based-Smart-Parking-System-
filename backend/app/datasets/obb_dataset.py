"""Convert prepared parking-space polygons into an oriented-bounding-box dataset.

The V2 localizer recognised one of twelve registered camera identities and
recalled that camera's stored polygons, so it could not represent a lot it had
never seen.  Training a detector instead requires the polygons expressed as
per-image labels, which is what this module produces.

Split design is the point of the exercise.  A random image split would let the
same camera appear in training and evaluation and would report a generalisation
number the deployment could never reproduce, so the splits here are defined by
*camera*:

``train``        cameras the detector is fitted on
``val_known``    held-out images from those same cameras
``val_unseen``   cameras that never appear in training -- the selection signal
``test_unseen``  a second disjoint set of unseen cameras, reporting only

``val_known`` versus ``val_unseen`` is the comparison that says whether the
detector has learned "what a parking space looks like" or merely "what these
particular car parks look like".
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

from app.datasets.integrity import deterministic_score

# Fraction of each training camera's images reserved to measure known-camera
# performance.  Held out by a deterministic hash so the split is reproducible.
KNOWN_HOLDOUT_FRACTION = 0.15

# Longest edge kept for training copies.  Comfortably above the 1024 training
# size while keeping per-image decode cost bounded.
MAX_IMAGE_SIDE = 1600

SPLITS = ("train", "val_known", "val_unseen", "test_unseen")
SINGLE_CLASS_NAMES = ("parking-space",)
OCCUPANCY_CLASS_NAMES = ("vacant", "occupied")


@dataclass
class ConversionReport:
    images: dict[str, int] = field(default_factory=dict)
    instances: dict[str, int] = field(default_factory=dict)
    cameras: dict[str, list[str]] = field(default_factory=dict)
    skipped_polygons: int = 0
    label_mode: str = "single"

    def as_dict(self) -> dict[str, object]:
        return {
            "images": self.images,
            "instances": self.instances,
            "cameras": self.cameras,
            "skipped_polygons": self.skipped_polygons,
            "label_mode": self.label_mode,
            "known_holdout_fraction": KNOWN_HOLDOUT_FRACTION,
        }


def camera_identity(row: dict[str, object]) -> str:
    """Stable identifier for the viewpoint a source image was taken from.

    PKLot and CNRPark group ids are ``site/date`` or ``camera/date``, so the
    first segment is the fixed camera.  Every ACPDS capture is its own
    viewpoint, which is what makes that dataset the main source of viewpoint
    diversity here.
    """
    dataset = str(row["dataset"])
    if dataset == "ACPDS":
        return f"ACPDS:{row['source_id']}"
    return f"{dataset}:{str(row['group_id']).split('/', 1)[0]}"


def assign_split(row: dict[str, object]) -> str:
    """Place a source image into one of the four camera-aware splits.

    ``val_known`` must contain only cameras that are also fitted on, otherwise
    it measures unseen-camera performance while claiming to measure known-camera
    performance.  Every ACPDS capture is a distinct viewpoint with a single
    image, so ACPDS never contributes to ``val_known`` -- holding one of its
    images out would remove that viewpoint from training entirely.
    """
    partition = str(row.get("localization_partition", "train"))
    if partition == "validation":
        return "val_unseen"
    if partition == "holdout":
        return "test_unseen"
    if str(row["dataset"]) == "ACPDS":
        return "train"
    # Deterministically reserve a slice of each multi-image camera's frames so
    # known-camera and unseen-camera accuracy can be compared directly.
    score = deterministic_score(str(row["source_id"])) % 1000
    return "val_known" if score < KNOWN_HOLDOUT_FRACTION * 1000 else "train"


def polygon_to_obb(polygon: Iterable[Iterable[float]]) -> list[float] | None:
    """Flatten a normalised quadrilateral into YOLO-OBB corner order.

    Returns ``None`` for anything that is not a usable quadrilateral, so a
    malformed annotation is dropped and counted rather than written as a
    degenerate box.
    """
    points = [[float(x), float(y)] for x, y in polygon]
    if len(points) != 4:
        return None
    flat: list[float] = []
    for x, y in points:
        if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
            return None
        flat.extend((round(x, 6), round(y, 6)))
    xs = flat[0::2]
    ys = flat[1::2]
    if max(xs) - min(xs) < 1e-4 or max(ys) - min(ys) < 1e-4:
        return None
    return flat


def build_dataset(
    source_manifest: Path,
    output_root: Path,
    *,
    label_mode: str = "single",
    link_images: bool = True,
) -> ConversionReport:
    """Write a YOLO-OBB dataset laid out as ``images/<split>`` + ``labels/<split>``.

    ``label_mode`` selects the experiment: ``single`` gives one
    ``parking-space`` class for the two-stage pipeline, ``occupancy`` gives
    ``vacant``/``occupied`` for the single-stage variant.
    """
    if label_mode not in {"single", "occupancy"}:
        raise ValueError("label_mode must be 'single' or 'occupancy'")
    rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines()]
    # Manifest image paths are stored relative to the manifest's own directory.
    data_root = source_manifest.parent

    report = ConversionReport(label_mode=label_mode)
    cameras: dict[str, set[str]] = {split: set() for split in SPLITS}
    counts = dict.fromkeys(SPLITS, 0)
    instances = dict.fromkeys(SPLITS, 0)

    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for row in rows:
        split = assign_split(row)
        stem = str(row["source_id"]).replace("/", "__").replace(" ", "_")
        source_image = data_root / str(row["image_path"])
        if not source_image.is_file():
            continue

        lines: list[str] = []
        for slot in row["slots"]:
            corners = polygon_to_obb(slot["polygon"])
            if corners is None:
                report.skipped_polygons += 1
                continue
            class_id = 0
            if label_mode == "occupancy":
                class_id = 1 if bool(slot.get("occupied")) else 0
            lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in corners))
        if not lines:
            continue

        (output_root / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        destination = output_root / "images" / split / f"{stem}.jpg"
        if link_images and not destination.exists():
            _link_or_copy(source_image, destination)

        counts[split] += 1
        instances[split] += len(lines)
        cameras[split].add(camera_identity(row))

    report.images = counts
    report.instances = instances
    report.cameras = {split: sorted(values) for split, values in cameras.items()}
    _write_data_yaml(output_root, label_mode)
    (output_root / "conversion-report.json").write_text(
        json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return report


def _link_or_copy(source: Path, destination: Path, max_side: int = MAX_IMAGE_SIDE) -> None:
    """Materialise a training copy of the source image, downscaled if oversized.

    ACPDS frames are 4000x3000.  At a 1024-pixel training size those pixels are
    discarded anyway, but every dataloader worker still decodes 36 MB per image,
    which exhausts host memory during mosaic augmentation.  Polygons are stored
    normalised, so resizing leaves the labels untouched.
    """
    from PIL import Image

    with Image.open(source) as image:
        if max(image.size) <= max_side:
            try:
                destination.hardlink_to(source)
                return
            except (OSError, NotImplementedError):
                destination.write_bytes(source.read_bytes())
                return
        scale = max_side / max(image.size)
        resized = image.convert("RGB").resize(
            (round(image.width * scale), round(image.height * scale)), Image.LANCZOS
        )
        resized.save(destination, "JPEG", quality=92)


def _write_data_yaml(output_root: Path, label_mode: str) -> None:
    names = OCCUPANCY_CLASS_NAMES if label_mode == "occupancy" else SINGLE_CLASS_NAMES
    lines = [
        f"path: {output_root.as_posix()}",
        "train: images/train",
        "val: images/val_unseen",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (output_root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # A second descriptor pointed at the known-camera split, so the same
    # weights can be evaluated against both populations without editing YAML.
    known = list(lines)
    known[2] = "val: images/val_known"
    (output_root / "data-known.yaml").write_text("\n".join(known) + "\n", encoding="utf-8")

    test = list(lines)
    test[2] = "val: images/test_unseen"
    (output_root / "data-test.yaml").write_text("\n".join(test) + "\n", encoding="utf-8")


def unseen_camera_overlap(report: ConversionReport) -> set[str]:
    """Cameras appearing in both training and an unseen split.

    Must be empty; a non-empty result means the split leaked and any
    unseen-camera metric computed from it is meaningless.
    """
    train = set(report.cameras.get("train", []))
    unseen = set(report.cameras.get("val_unseen", [])) | set(
        report.cameras.get("test_unseen", [])
    )
    return train & unseen


def known_camera_leak(report: ConversionReport) -> set[str]:
    """Cameras in ``val_known`` that were never trained on.

    Must be empty; anything here would be an unseen camera masquerading as a
    known one and would understate the known/unseen gap.
    """
    return set(report.cameras.get("val_known", [])) - set(report.cameras.get("train", []))
