from __future__ import annotations

import argparse
import json
from pathlib import Path

import imagehash
from PIL import Image

from app.core.config import get_settings
from app.datasets.integrity import perceptual_distance, read_jsonl, sha256_file


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Verify a separately sourced, authorized final confirmation set"
    )
    parser.add_argument("confirmation_root", type=Path)
    parser.add_argument("--protocol-root", type=Path, default=settings.parking_data_root)
    arguments = parser.parse_args()
    root = arguments.confirmation_root.resolve()
    manifest_path = root / "confirmation-manifest.jsonl"
    rows = read_jsonl(manifest_path)
    if not rows:
        raise RuntimeError("External confirmation manifest is empty")
    protocol_sources = read_jsonl(
        arguments.protocol_root / "prepared" / "v2-protocol" / "source-manifest.jsonl"
    )
    known_exact = {str(row["content_sha256"]) for row in protocol_sources}
    known_perceptual = [str(row["perceptual_hash"]) for row in protocol_sources]
    checked = []
    for row in rows:
        required = {
            "id",
            "dataset",
            "image_path",
            "source_sha256",
            "group_id",
            "slots",
            "license_or_authorization",
            "provenance",
        }
        missing = required - set(row)
        if missing:
            raise RuntimeError(f"Confirmation row is incomplete: {sorted(missing)}")
        image_path = (root / str(row["image_path"])).resolve()
        if root not in image_path.parents or not image_path.is_file():
            raise RuntimeError(f"Confirmation image is missing or unsafe: {row['id']}")
        digest = sha256_file(image_path)
        if digest != row["source_sha256"] or digest in known_exact:
            raise RuntimeError(
                f"Confirmation image is changed or overlaps development data: {row['id']}"
            )
        with Image.open(image_path) as image:
            perceptual = str(imagehash.phash(image.convert("RGB")))
        if any(perceptual_distance(perceptual, known) <= 3 for known in known_perceptual):
            raise RuntimeError(
                f"Confirmation image is a near duplicate of development data: {row['id']}"
            )
        checked.append(str(row["id"]))
    print(
        json.dumps(
            {
                "valid": True,
                "status": "authorized_external_confirmation_ready_for_single_evaluation",
                "samples": len(checked),
                "manifest_sha256": sha256_file(manifest_path),
                "ids": checked,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
