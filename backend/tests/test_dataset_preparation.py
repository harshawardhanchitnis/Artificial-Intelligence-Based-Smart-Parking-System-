from __future__ import annotations

import csv
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from app.datasets.preparation import (
    PreparationError,
    prepare_demo,
    safe_extract_zip,
    verify_prepared_data,
)
from app.services.catalogue_service import CatalogueRepository


def jpeg(width: int = 100, height: int = 80, color: str = "white") -> bytes:
    stream = io.BytesIO()
    image = Image.new("RGB", (width, height), "white")
    if color == "blue":
        ImageDraw.Draw(image).rectangle((5, 5, 35, 70), fill="blue")
    elif color == "red":
        ImageDraw.Draw(image).polygon(((50, 5), (95, 40), (50, 75)), fill="red")
    image.save(stream, "JPEG")
    return stream.getvalue()


def add_tar_bytes(archive: tarfile.TarFile, name: str, payload: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(payload)
    archive.addfile(info, io.BytesIO(payload))


def build_archives(data_root: Path) -> None:
    pklot = data_root / "archives" / "PKLot" / "PKLot.tar.gz"
    pklot.parent.mkdir(parents=True)
    with tarfile.open(pklot, "w:gz") as archive:
        base = "PKLot/PKLot/UFPR04/Sunny/2012-12-17/2012-12-17_06_05_00"
        add_tar_bytes(archive, f"{base}.jpg", jpeg())
        xml = (
            b'<parking><space id="1" occupied="1"><contour>'
            b'<point x="10" y="10"/><point x="40" y="10"/>'
            b'<point x="40" y="50"/><point x="10" y="50"/>'
            b"</contour></space></parking>"
        )
        add_tar_bytes(archive, f"{base}.xml", xml)

    cnr_root = data_root / "archives" / "CNRPark+EXT"
    cnr_root.mkdir(parents=True)
    with tarfile.open(cnr_root / "CNR-EXT_FULL_IMAGE_1000x750.tar", "w") as archive:
        add_tar_bytes(
            archive, "FULL_IMAGE_1000x750/camera1.csv", b"SlotId,X,Y,W,H\n1,100,100,200,300\n"
        )
        add_tar_bytes(
            archive,
            "FULL_IMAGE_1000x750/SUNNY/2015-11-12/camera1/2015-11-12_0709.jpg",
            jpeg(color="blue"),
        )
    with (cnr_root / "CNRPark+EXT.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["camera", "datetime", "occupancy", "slot_id"])
        writer.writeheader()
        writer.writerow(
            {"camera": "01", "datetime": "2015-11-12_07.09", "occupancy": "0", "slot_id": "1"}
        )
    for name in ("CNR-EXT-Patches-150x150.zip", "CNRPark-Patches-150x150.zip", "splits.zip"):
        with zipfile.ZipFile(cnr_root / name, "w") as archive:
            archive.writestr("placeholder.txt", "fixture")

    acpds = data_root / "archives" / "ACPDS" / "parking_rois_gopro.zip"
    acpds.parent.mkdir(parents=True)
    annotations = {
        "train": {
            "file_names": ["sample.JPG"],
            "rois_list": [[[[0.1, 0.1], [0.4, 0.1], [0.4, 0.7], [0.1, 0.7]]]],
            "occupancy_list": [[True]],
        },
        "valid": {"file_names": [], "rois_list": [], "occupancy_list": []},
        "test": {"file_names": [], "rois_list": [], "occupancy_list": []},
    }
    with zipfile.ZipFile(acpds, "w") as archive:
        archive.writestr("annotations.json", json.dumps(annotations))
        archive.writestr("images/sample.JPG", jpeg(color="red"))


def test_demo_preparation_builds_verified_catalogue(tmp_path: Path) -> None:
    build_archives(tmp_path)
    report = prepare_demo(tmp_path, samples_per_dataset=1, progress=None, validate_sizes=False)
    assert report["scenario_count"] == 3
    verification = verify_prepared_data(tmp_path)
    assert verification["valid"] is True
    assert verification["dataset_counts"] == {"ACPDS": 1, "CNRPark+EXT": 1, "PKLot": 1}

    repository = CatalogueRepository(tmp_path)
    scenarios = repository.list_scenarios()
    assert len(scenarios) == 3
    assert repository.image_path(str(scenarios[0]["id"])).is_file()


def test_safe_zip_extraction_rejects_path_traversal(tmp_path: Path) -> None:
    archive_path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("../escape.txt", "not allowed")
    with pytest.raises(PreparationError, match="Unsafe archive member"):
        safe_extract_zip(archive_path, tmp_path / "output")
    assert not (tmp_path / "escape.txt").exists()


def test_safe_zip_extraction_rejects_symbolic_links(tmp_path: Path) -> None:
    archive_path = tmp_path / "link.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        member = zipfile.ZipInfo("shortcut")
        member.create_system = 3
        member.external_attr = 0o120777 << 16
        archive.writestr(member, "target.txt")
    with pytest.raises(PreparationError, match="Links are not allowed"):
        safe_extract_zip(archive_path, tmp_path / "output")
