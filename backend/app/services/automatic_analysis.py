from __future__ import annotations

import json
import uuid
from pathlib import Path
from time import perf_counter

from PIL import Image, ImageDraw, ImageFont
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import (
    AnalysisRecord,
    DetectedLayout,
    DetectedSlot,
    LayoutCorrection,
    MediaAsset,
)
from app.ml.geometry import rectify_slot
from app.ml.localization import SlotLocalizerPredictor
from app.ml.occupancy_v3 import OccupancyV3Predictor
from app.services.media_service import resolve_media_path


def _overlay(image: Image.Image, predictions: list[dict[str, object]], destination: Path) -> None:
    canvas = image.convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = canvas.size
    for index, prediction in enumerate(predictions, 1):
        polygon = [(round(x * width), round(y * height)) for x, y in prediction["polygon"]]
        occupied = bool(prediction["predicted_occupied"])
        color = (220, 38, 38, 210) if occupied else (22, 163, 74, 210)
        fill = (*color[:3], 45)
        draw.polygon(polygon, fill=fill, outline=color, width=max(2, width // 600))
        center = (
            round(sum(point[0] for point in polygon) / len(polygon)),
            round(sum(point[1] for point in polygon) / len(polygon)),
        )
        draw.text(center, str(index), fill=(255, 255, 255, 255), font=ImageFont.load_default())
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(canvas, layer).convert("RGB").save(destination, "JPEG", quality=92)


def classify_layout(
    image: Image.Image,
    slots: list[dict[str, object]],
    model_root: Path,
) -> tuple[list[dict[str, object]], float]:
    classifier = OccupancyV3Predictor.load(model_root)
    patches = [rectify_slot(image, slot["polygon"]) for slot in slots]
    states, elapsed = classifier.predict(patches)
    predictions = []
    for slot, state in zip(slots, states, strict=True):
        predictions.append({**slot, **state})
    return predictions, elapsed


def persist_image_analysis(
    session: Session,
    settings: Settings,
    media: MediaAsset,
    *,
    corrected_slots: list[dict[str, object]] | None = None,
    correction_of_layout_id: int | None = None,
    correction_note: str | None = None,
) -> dict[str, object]:
    started = perf_counter()
    path = resolve_media_path(settings, media.storage_path)
    with Image.open(path) as source:
        image = source.convert("RGB")
        if corrected_slots is None:
            localization = SlotLocalizerPredictor.load(settings.model_root).detect(image)
            slots = localization["slots"]
        else:
            localization = {
                "status": "success_with_warnings",
                "confidence": 1.0,
                "model_name": "advanced-user-correction",
            }
            slots = corrected_slots
        layout = DetectedLayout(
            media_asset_id=media.id,
            model_name=str(localization["model_name"]),
            layout_identifier=uuid.uuid4().hex,
            confidence=float(localization["confidence"]),
            status="corrected" if corrected_slots is not None else "automatic",
            geometry_json=json.dumps(slots, separators=(",", ":")),
        )
        session.add(layout)
        session.flush()
        for index, slot in enumerate(slots, 1):
            session.add(
                DetectedSlot(
                    layout_id=layout.id,
                    slot_index=index,
                    polygon_json=json.dumps(slot["polygon"], separators=(",", ":")),
                    localization_confidence=float(slot.get("localization_confidence", 1.0)),
                    corner_confidence=float(slot.get("corner_confidence", 1.0)),
                )
            )
        if corrected_slots is not None and correction_of_layout_id is not None:
            original = session.get(DetectedLayout, correction_of_layout_id)
            if original is None or original.media_asset_id != media.id:
                raise ValueError("Correction source layout is invalid for this media asset")
            session.add(
                LayoutCorrection(
                    layout_id=layout.id,
                    before_json=original.geometry_json,
                    after_json=layout.geometry_json,
                    note=correction_note or "Advanced correction of automatically detected layout",
                )
            )
        if localization["status"] == "unsupported_layout":
            session.commit()
            return {
                "status": "unsupported_layout",
                "message": (
                    "Automatic localisation confidence is insufficient for a trustworthy result."
                ),
                "media_asset_id": media.id,
                "layout_id": layout.id,
                "localization_confidence": localization["confidence"],
                "detected_spaces": len(slots),
                "advanced_correction_available": True,
            }
        predictions, inference_ms = classify_layout(image, slots, settings.model_root)
        occupied = sum(bool(value["predicted_occupied"]) for value in predictions)
        relative_result = Path("media") / "results" / "images" / f"{uuid.uuid4().hex}.jpg"
        result_path = settings.parking_data_root / relative_result
        _overlay(image, predictions, result_path)
        record = AnalysisRecord(
            dataset="User upload",
            scenario_id=f"upload:{media.sha256[:20]}",
            total_spaces=len(predictions),
            occupied_spaces=occupied,
            vacant_spaces=len(predictions) - occupied,
            processing_time_ms=(perf_counter() - started) * 1_000,
            result_image_path=relative_result.as_posix(),
            model_name=OccupancyV3Predictor.load(settings.model_root).metadata["model_name"],
            average_confidence=sum(float(value["confidence"]) for value in predictions)
            / len(predictions),
            prediction_json=json.dumps(predictions, separators=(",", ":")),
            source_type="image_upload",
            media_asset_id=media.id,
            layout_id=layout.id,
            localization_confidence=float(localization["confidence"]),
            result_status=str(localization["status"]),
        )
        session.add(record)
        session.commit()
        session.refresh(record)
    return {
        "status": record.result_status,
        "analysis_id": record.id,
        "media_asset_id": media.id,
        "layout_id": layout.id,
        "correction_of_layout_id": correction_of_layout_id,
        "total_spaces": record.total_spaces,
        "occupied_spaces": record.occupied_spaces,
        "vacant_spaces": record.vacant_spaces,
        "average_confidence": round(float(record.average_confidence or 0), 6),
        "localization_confidence": round(float(record.localization_confidence or 0), 6),
        "processing_time_ms": round(record.processing_time_ms, 3),
        "inference_time_ms": round(inference_ms, 3),
        "predictions": predictions,
        "result_image_url": f"/api/v1/media/analyses/{record.id}/image",
    }
