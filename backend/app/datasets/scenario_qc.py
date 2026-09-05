from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from app.datasets.integrity import atomic_json, sha256_file
from app.ml.geometry import order_polygon, validate_polygon


def generate_scenario_qc(data_root: Path, *, reviewer: str | None = None) -> dict[str, object]:
    catalogue_path = data_root / "demo" / "catalogue.json"
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    destination = data_root / "prepared" / "scenario-qc"
    destination.mkdir(parents=True, exist_ok=True)
    inventory = []
    sheets = {}
    for dataset in ("PKLot", "CNRPark+EXT", "ACPDS"):
        thumbnails = []
        scenarios = [row for row in catalogue["scenarios"] if row["dataset"] == dataset]
        for scenario in scenarios:
            image_path = data_root / str(scenario["image_path"])
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            draw = ImageDraw.Draw(image, "RGBA")
            width, height = image.size
            invalid = 0
            for index, slot in enumerate(scenario["slots"], 1):
                polygon = order_polygon(slot["polygon"])
                quality = validate_polygon(polygon)
                invalid += int(not quality.valid)
                points = [(round(x * width), round(y * height)) for x, y in polygon]
                color = (220, 38, 38, 210) if slot["occupied"] else (5, 150, 105, 210)
                draw.polygon(points, fill=(*color[:3], 35), outline=color, width=2)
                center = tuple(round(sum(point[axis] for point in points) / 4) for axis in (0, 1))
                draw.text(
                    center, str(index), fill=(255, 255, 255, 255), font=ImageFont.load_default()
                )
            overlay_path = destination / f"{scenario['id']}.jpg"
            image.save(overlay_path, "JPEG", quality=90)
            thumb = image.copy()
            thumb.thumbnail((560, 360))
            card = Image.new("RGB", (600, 410), "white")
            card.paste(thumb, ((600 - thumb.width) // 2, 8))
            ImageDraw.Draw(card).text(
                (12, 375),
                f"{scenario['id']} | {len(scenario['slots'])} slots | {scenario['condition']}",
                fill="black",
            )
            thumbnails.append(card)
            inventory.append(
                {
                    "scenario_id": scenario["id"],
                    "dataset": dataset,
                    "condition": scenario["condition"],
                    "slot_count": len(scenario["slots"]),
                    "invalid_geometry": invalid,
                    "overlay_path": overlay_path.relative_to(data_root).as_posix(),
                    "overlay_sha256": sha256_file(overlay_path),
                }
            )
        sheet = Image.new("RGB", (1200, max(1, (len(thumbnails) + 1) // 2) * 410), "#e2e8f0")
        for index, thumbnail in enumerate(thumbnails):
            sheet.paste(thumbnail, ((index % 2) * 600, (index // 2) * 410))
        sheet_path = destination / f"{dataset.lower().replace('+', '-')}-contact-sheet.jpg"
        sheet.save(sheet_path, "JPEG", quality=90)
        sheets[dataset] = {
            "path": sheet_path.relative_to(data_root).as_posix(),
            "sha256": sha256_file(sheet_path),
        }
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "catalogue_sha256": sha256_file(catalogue_path),
        "scenario_count": len(inventory),
        "geometry_valid": all(row["invalid_geometry"] == 0 for row in inventory),
        "manual_overlay_qc": {
            "status": "confirmed" if reviewer else "review_required",
            "reviewer": reviewer,
            "confirmed_at": datetime.now(UTC).isoformat() if reviewer else None,
            "scope": "all generated dataset contact sheets",
        },
        "contact_sheets": sheets,
        "scenarios": inventory,
    }
    atomic_json(destination / "scenario-qc-report.json", report)
    return report
