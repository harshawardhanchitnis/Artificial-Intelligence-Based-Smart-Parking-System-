"""Build parking-space datasets that keep the true four-corner bay geometry.

The oriented-bounding-box dataset flattens each bay into a rotated rectangle
because that is all an OBB head can emit.  Measured on the prepared corpus,
that costs a mean 0.907 polygon IoU on unseen cameras and 0.762 on wide-angle
ACPDS views before a model has been trained at all -- a ceiling no amount of
training can lift.  Perspective turns a rectangular bay on the ground into a
general quadrilateral in the image, so the representation has to follow.

Three label formats are written from the same manifest and the same
camera-aware splits, so the architectures that consume them are compared on
identical data:

``pose``  axis-aligned box plus four independently regressed corners
``seg``   the quadrilateral as an instance mask, corners recovered from contour
``obb``   the incumbent rotated rectangle, kept as the control

A second axis selects which sources contribute.  Every CNRPark+EXT annotation
in the corpus is an *axis-aligned* rectangle rather than a bay outline -- 62,668
of 135,802 instances, 46% of the corpus -- so those labels teach a geometry
model that parking spaces are upright boxes.  ``geometry_sources_only`` drops
them, and the two variants are trained and compared rather than assumed.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from app.datasets.obb_dataset import (
    KNOWN_HOLDOUT_FRACTION,
    MAX_IMAGE_SIDE,
    SPLITS,
    assign_split,
    camera_identity,
)
from app.ml.geometry import order_polygon

LABEL_FORMATS = ("pose", "seg", "obb")

# Sources whose polygons follow the real bay outline.  CNRPark+EXT is excluded
# from this set because its annotations are upright rectangles; it stays
# available for occupancy work, where the crop is what matters.
GEOMETRY_FAITHFUL_DATASETS = ("PKLot", "ACPDS")

# Corners are stored clockwise from the top-left, so a horizontal flip maps
# top-left to top-right and bottom-right to bottom-left.
POSE_FLIP_INDEX = (1, 0, 3, 2)

# Smallest normalised extent a bay may span before it is treated as degenerate.
MIN_EXTENT = 1e-4


@dataclass
class QuadConversionReport:
    label_format: str = "pose"
    geometry_sources_only: bool = False
    images: dict[str, int] = field(default_factory=dict)
    instances: dict[str, int] = field(default_factory=dict)
    cameras: dict[str, list[str]] = field(default_factory=dict)
    datasets: dict[str, dict[str, int]] = field(default_factory=dict)
    skipped_polygons: int = 0

    def as_dict(self) -> dict[str, object]:
        return {
            "label_format": self.label_format,
            "geometry_sources_only": self.geometry_sources_only,
            "geometry_faithful_datasets": list(GEOMETRY_FAITHFUL_DATASETS),
            "images": self.images,
            "instances": self.instances,
            "datasets": self.datasets,
            "cameras": self.cameras,
            "skipped_polygons": self.skipped_polygons,
            "known_holdout_fraction": KNOWN_HOLDOUT_FRACTION,
        }


def canonical_quad(polygon: Iterable[Iterable[float]]) -> list[list[float]] | None:
    """Order a bay's four corners clockwise from the top-left, or reject it.

    Keypoint regression only works if corner *one* means the same physical
    corner in every image, so the ordering has to be a property of the shape
    and the image frame rather than of the order the annotator happened to
    click.  Rejecting instead of repairing keeps malformed annotations out of
    the training signal and counted in the report.
    """
    points = [[float(x), float(y)] for x, y in polygon]
    if len(points) != 4:
        return None
    if any(not (0.0 <= value <= 1.0) for point in points for value in point):
        return None
    ordered = order_polygon(points)
    if len(ordered) != 4:
        return None
    xs = [point[0] for point in ordered]
    ys = [point[1] for point in ordered]
    if max(xs) - min(xs) < MIN_EXTENT or max(ys) - min(ys) < MIN_EXTENT:
        return None
    return ordered


def pose_label(quad: Sequence[Sequence[float]]) -> str:
    """One YOLO pose line: enclosing box, then four corners marked visible.

    Corners are marked visible even where a parked vehicle hides them.  A bay
    boundary is a property of the ground, not of what is standing on it, and a
    product that only reported corners it could literally see would lose the
    bay the moment it filled up.
    """
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    values = [(min_x + max_x) / 2, (min_y + max_y) / 2, max_x - min_x, max_y - min_y]
    for x, y in quad:
        values.extend((x, y, 2.0))
    return "0 " + " ".join(f"{value:.6f}" for value in values)


def seg_label(quad: Sequence[Sequence[float]]) -> str:
    """One YOLO segmentation line: the quadrilateral itself as a mask."""
    flat = [value for point in quad for value in point]
    return "0 " + " ".join(f"{value:.6f}" for value in flat)


def obb_label(quad: Sequence[Sequence[float]]) -> str:
    """One YOLO-OBB line, kept so the incumbent trains on identical splits."""
    flat = [value for point in quad for value in point]
    return "0 " + " ".join(f"{value:.6f}" for value in flat)


_WRITERS = {"pose": pose_label, "seg": seg_label, "obb": obb_label}


def build_quad_dataset(
    source_manifest: Path,
    output_root: Path,
    *,
    label_format: str = "pose",
    geometry_sources_only: bool = True,
    link_images: bool = True,
) -> QuadConversionReport:
    """Write ``images/<split>`` + ``labels/<split>`` in the requested format."""
    if label_format not in _WRITERS:
        raise ValueError(f"label_format must be one of {LABEL_FORMATS}")
    rows = [json.loads(line) for line in source_manifest.read_text(encoding="utf-8").splitlines()]
    data_root = source_manifest.parent
    writer = _WRITERS[label_format]

    report = QuadConversionReport(
        label_format=label_format, geometry_sources_only=geometry_sources_only
    )
    cameras: dict[str, set[str]] = {split: set() for split in SPLITS}
    counts = dict.fromkeys(SPLITS, 0)
    instances = dict.fromkeys(SPLITS, 0)
    datasets: dict[str, dict[str, int]] = {}

    for split in SPLITS:
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    for row in rows:
        dataset = str(row["dataset"])
        if geometry_sources_only and dataset not in GEOMETRY_FAITHFUL_DATASETS:
            continue
        split = assign_split(row)
        stem = str(row["source_id"]).replace("/", "__").replace(" ", "_")
        source_image = data_root / str(row["image_path"])
        if not source_image.is_file():
            continue

        lines: list[str] = []
        for slot in row["slots"]:
            quad = canonical_quad(slot["polygon"])
            if quad is None:
                report.skipped_polygons += 1
                continue
            lines.append(writer(quad))
        if not lines:
            continue

        (output_root / "labels" / split / f"{stem}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        destination = output_root / "images" / split / f"{stem}.jpg"
        if link_images and not destination.exists():
            _materialise(source_image, destination)

        counts[split] += 1
        instances[split] += len(lines)
        cameras[split].add(camera_identity(row))
        bucket = datasets.setdefault(dataset, dict.fromkeys(SPLITS, 0))
        bucket[split] += len(lines)

    report.images = counts
    report.instances = instances
    report.datasets = datasets
    report.cameras = {split: sorted(values) for split, values in cameras.items()}
    _write_data_yaml(output_root, label_format)
    (output_root / "conversion-report.json").write_text(
        json.dumps(report.as_dict(), indent=2) + "\n", encoding="utf-8"
    )
    return report


def _materialise(source: Path, destination: Path, max_side: int = MAX_IMAGE_SIDE) -> None:
    """Copy a training image, downscaling oversized frames.

    Polygons are stored normalised, so resizing leaves every label untouched.
    """
    from PIL import Image

    with Image.open(source) as handle:
        image = handle.convert("RGB")
        if max(image.size) > max_side:
            scale = max_side / max(image.size)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.LANCZOS,
            )
        image.save(destination, "JPEG", quality=92)


def _write_data_yaml(output_root: Path, label_format: str) -> None:
    """Emit one descriptor per evaluation split.

    Ultralytics validates whatever a descriptor calls ``val``, so known-camera
    and unseen-camera accuracy need separate files rather than separate runs
    over one file.
    """
    root = str(output_root).replace("\\", "/")
    pose_header = ""
    if label_format == "pose":
        flip = ", ".join(str(index) for index in POSE_FLIP_INDEX)
        pose_header = f"kpt_shape: [4, 3]\nflip_idx: [{flip}]\n"
    for name, split in (
        ("data.yaml", "val_unseen"),
        ("data-known.yaml", "val_known"),
        ("data-test.yaml", "test_unseen"),
    ):
        (output_root / name).write_text(
            f"path: {root}\n"
            f"train: images/train\n"
            f"val: images/{split}\n"
            f"{pose_header}"
            f"names:\n  0: parking-space\n",
            encoding="utf-8",
        )
