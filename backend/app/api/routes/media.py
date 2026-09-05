from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import AnalysisRecord, DetectedLayout, DetectedSlot, MediaAsset
from app.db.session import SessionLocal
from app.ml.geometry import order_polygon, validate_polygon
from app.ml.localization import LocalizerNotReadyError
from app.ml.occupancy_v3 import OccupancyV3NotReadyError
from app.services.automatic_analysis import persist_image_analysis
from app.services.media_service import (
    MediaValidationError,
    resolve_media_path,
    safe_display_name,
    store_image,
)

router = APIRouter()


class CorrectedSlot(BaseModel):
    polygon: list[list[float]] = Field(min_length=4, max_length=12)


class LayoutCorrectionRequest(BaseModel):
    slots: list[CorrectedSlot] = Field(min_length=1, max_length=300)
    note: str | None = Field(default=None, max_length=500)


@router.post("/images/analyse")
async def analyse_uploaded_image(
    file: UploadFile = File(...),  # noqa: B008
) -> dict[str, object]:
    settings = get_settings()
    try:
        stored = await store_image(file, settings)
    except MediaValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    with SessionLocal() as session:
        media = MediaAsset(
            media_type="image",
            original_name=safe_display_name(file.filename, "parking-image.jpg"),
            storage_path=stored.relative_path,
            sha256=stored.sha256,
            mime_type=stored.mime_type,
            size_bytes=stored.size_bytes,
            width=stored.width,
            height=stored.height,
        )
        existing = session.scalar(
            select(MediaAsset).where(
                MediaAsset.media_type == "image", MediaAsset.sha256 == stored.sha256
            )
        )
        if existing is not None:
            resolve_media_path(settings, stored.relative_path).unlink(missing_ok=True)
            media = existing
        else:
            session.add(media)
            session.commit()
            session.refresh(media)
        try:
            return persist_image_analysis(session, settings, media)
        except (LocalizerNotReadyError, OccupancyV3NotReadyError) as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc


@router.post("/layouts/{layout_id}/corrections")
def correct_detected_layout(layout_id: int, request: LayoutCorrectionRequest) -> dict[str, object]:
    settings = get_settings()
    slots = []
    for index, supplied in enumerate(request.slots, 1):
        polygon = order_polygon(supplied.polygon)
        quality = validate_polygon(polygon)
        if not quality.valid:
            raise HTTPException(
                status_code=422, detail=f"Slot {index} is invalid: {quality.reason}"
            )
        slots.append(
            {
                "id": str(index),
                "polygon": polygon,
                "localization_confidence": 1.0,
                "corner_confidence": 1.0,
            }
        )
    with SessionLocal() as session:
        layout = session.get(DetectedLayout, layout_id)
        if layout is None:
            raise HTTPException(status_code=404, detail="Detected layout not found")
        media = session.get(MediaAsset, layout.media_asset_id)
        if media is None:
            raise HTTPException(status_code=404, detail="Source media not found")
        return persist_image_analysis(
            session,
            settings,
            media,
            corrected_slots=slots,
            correction_of_layout_id=layout_id,
            correction_note=request.note,
        )


@router.get("/layouts/{layout_id}")
def detected_layout(layout_id: int) -> dict[str, object]:
    with SessionLocal() as session:
        layout = session.get(DetectedLayout, layout_id)
        if layout is None:
            raise HTTPException(status_code=404, detail="Detected layout not found")
        slots = session.query(DetectedSlot).filter(DetectedSlot.layout_id == layout.id).all()
        return {
            "id": layout.id,
            "media_asset_id": layout.media_asset_id,
            "layout_identifier": layout.layout_identifier,
            "model_name": layout.model_name,
            "confidence": layout.confidence,
            "status": layout.status,
            "reference_frame_seconds": layout.reference_frame_seconds,
            "slots": [
                {
                    "id": slot.id,
                    "slot_index": slot.slot_index,
                    "polygon": json.loads(slot.polygon_json),
                    "localization_confidence": slot.localization_confidence,
                    "corner_confidence": slot.corner_confidence,
                }
                for slot in slots
            ],
            "created_at": layout.created_at,
        }


@router.get("/assets/{media_id}/metadata")
def media_metadata(media_id: int) -> dict[str, object]:
    with SessionLocal() as session:
        media = session.get(MediaAsset, media_id)
        if media is None:
            raise HTTPException(status_code=404, detail="Media asset not found")
        return {
            "id": media.id,
            "media_type": media.media_type,
            "original_name": media.original_name,
            "sha256": media.sha256,
            "mime_type": media.mime_type,
            "size_bytes": media.size_bytes,
            "width": media.width,
            "height": media.height,
            "duration_seconds": media.duration_seconds,
            "fps": media.fps,
            "frame_count": media.frame_count,
            "created_at": media.created_at,
        }


@router.get("/analyses/{analysis_id}/image")
def analysis_image(analysis_id: int) -> FileResponse:
    settings = get_settings()
    with SessionLocal() as session:
        record = session.get(AnalysisRecord, analysis_id)
        if record is None or not record.result_image_path:
            raise HTTPException(status_code=404, detail="Result image not found")
        path = resolve_media_path(settings, record.result_image_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result image file not found")
    return FileResponse(
        path, media_type="image/jpeg", filename=f"parking-analysis-{analysis_id}.jpg"
    )


@router.get("/assets/{media_id}")
def source_media(media_id: int) -> FileResponse:
    settings = get_settings()
    with SessionLocal() as session:
        media = session.get(MediaAsset, media_id)
        if media is None:
            raise HTTPException(status_code=404, detail="Media asset not found")
        path = resolve_media_path(settings, media.storage_path)
        mime = media.mime_type
        name = media.original_name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Media file not found")
    return FileResponse(path, media_type=mime, filename=Path(name).name)
