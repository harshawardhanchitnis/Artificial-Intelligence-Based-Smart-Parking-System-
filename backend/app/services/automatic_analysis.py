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
from app.ml.camera_fingerprint import camera_signature
from app.ml.geometry import rectify_slots
from app.ml.occupancy_v3 import OccupancyV3Predictor
from app.ml.parking_area import AREA_UNKNOWN
from app.services.localization_service import (
    USER_VERIFIED,
    LayoutResolution,
    resolve_layout,
)
from app.services.media_service import resolve_media_path
from app.services.scene_analysis import (
    PRODUCT_MODE,
    SceneAnalysis,
    analyse_product,
    detect_vehicles_with_diagnostics,
)

# Bay outline colour per occupancy state.  Amber is a deliberate third colour
# rather than a shade of one of the others: an uncertain bay is a different
# answer from a vacant one, not a weaker version of it.
_STATE_COLOURS = {
    "occupied": (220, 38, 38, 210),
    "vacant": (22, 163, 74, 210),
    "uncertain": (217, 119, 6, 210),
}

# Vehicles are drawn in a different visual language from the bays: bays are
# filled polygons carrying a verdict colour, vehicles are unfilled boxes with a
# class chip.  The two must never be confusable, because they answer different
# questions -- "is this space free?" against "what did the detector actually
# find?".
_VEHICLE_COLOUR = (56, 189, 248, 235)
_UNMAPPED_COLOUR = (168, 85, 247, 235)

# A vehicle-shaped object the taxonomy does not cover.  It is drawn, because
# hiding it would make the overlay disagree with the occupancy decision it
# contributed to, but it is drawn faintly and carries **no class text**: the
# product exposes exactly CAR, TWO_WHEELER and TRUCK, and this object has earned
# none of them.
_UNCLASSIFIED_COLOUR = (148, 163, 184, 170)

# Candidate geometry shown when the layout was not established.  Deliberately
# neither green nor red: no occupancy verdict is being asserted about these.
_CANDIDATE_COLOUR = (100, 116, 139, 200)


def _label_font(width: int) -> ImageFont.ImageFont:
    """A legible label at any image size, from the bundled default face."""
    return ImageFont.load_default(size=max(11, min(28, round(width / 62))))


def _draw_vehicle_label(
    draw: ImageDraw.ImageDraw,
    text: str,
    anchor: tuple[int, int],
    colour: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
    bounds: tuple[int, int],
) -> None:
    """Draw a filled chip so the class reads over any background."""
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    text_width, text_height = right - left, bottom - top
    padding = max(2, text_height // 4)
    x = min(max(0, anchor[0]), bounds[0] - text_width - 2 * padding)
    y = anchor[1] - text_height - 2 * padding
    if y < 0:  # no room above the box, so sit just inside its top edge instead
        y = anchor[1] + padding
    draw.rectangle(
        [x, y, x + text_width + 2 * padding, y + text_height + 2 * padding],
        fill=(*colour[:3], 235),
    )
    draw.text((x + padding - left, y + padding - top), text, fill=(9, 12, 20, 255), font=font)


def _overlay(
    image: Image.Image,
    predictions: list[dict[str, object]],
    destination: Path,
    *,
    vehicles: list[dict[str, object]] | None = None,
    unmapped: set[int] | None = None,
    candidates: list[dict[str, object]] | None = None,
) -> None:
    """Render the analysed image: bay verdicts, and the vehicles behind them.

    Aggregate counts are not enough to judge a detection product.  A reader has
    to be able to see *where* the detector thought a vehicle was, so every
    supported vehicle is drawn with its class and confidence, and every bay
    keeps its verdict colour and index.  ``candidates`` draws proposed geometry
    in a neutral colour for the case where no layout was established -- shown as
    proposals, never as verdicts.
    """
    canvas = image.convert("RGBA")
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    width, height = canvas.size
    unmapped = unmapped or set()
    font = _label_font(width)
    vehicle_stroke = max(2, width // 500)

    for prediction in candidates or []:
        polygon = [(round(x * width), round(y * height)) for x, y in prediction["polygon"]]
        draw.polygon(
            polygon,
            fill=(*_CANDIDATE_COLOUR[:3], 30),
            outline=_CANDIDATE_COLOUR,
            width=max(2, width // 700),
        )

    for index, vehicle in enumerate(vehicles or []):
        x1, y1, x2, y2 = vehicle["box"]  # type: ignore[misc]
        box = [round(x1 * width), round(y1 * height), round(x2 * width), round(y2 * height)]
        product_class = vehicle.get("vehicle_class")
        if product_class is None:
            # Internal evidence only: visible, but never given a class name.
            draw.rectangle(box, outline=_UNCLASSIFIED_COLOUR, width=max(1, width // 900))
            continue
        colour = _UNMAPPED_COLOUR if index in unmapped else _VEHICLE_COLOUR
        draw.rectangle(box, outline=colour, width=vehicle_stroke)
        confidence = float(vehicle.get("confidence", 0.0))  # type: ignore[arg-type]
        _draw_vehicle_label(
            draw,
            f"{product_class} {confidence:.2f}",
            (box[0], box[1]),
            colour,
            font,
            (width, height),
        )

    for index, prediction in enumerate(predictions, 1):
        polygon = [(round(x * width), round(y * height)) for x, y in prediction["polygon"]]
        state = str(prediction.get("occupancy_state") or
                    ("occupied" if prediction.get("predicted_occupied") else "vacant"))
        color = _STATE_COLOURS.get(state, _STATE_COLOURS["vacant"])
        fill = (*color[:3], 45)
        draw.polygon(polygon, fill=fill, outline=color, width=max(2, width // 600))
        center = (
            round(sum(point[0] for point in polygon) / len(polygon)),
            round(sum(point[1] for point in polygon) / len(polygon)),
        )
        draw.text(center, str(index), fill=(255, 255, 255, 255), font=font, anchor="mm")
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.alpha_composite(canvas, layer).convert("RGB").save(destination, "JPEG", quality=92)


def classify_layout(
    image: Image.Image,
    slots: list[dict[str, object]],
    model_root: Path,
) -> tuple[list[dict[str, object]], float, str]:
    classifier = OccupancyV3Predictor.load(model_root)
    patches = rectify_slots(image, [slot["polygon"] for slot in slots])
    states, elapsed = classifier.predict(patches)
    predictions = []
    for slot, state in zip(slots, states, strict=True):
        predictions.append({**slot, **state})
    return predictions, elapsed, str(classifier.metadata["model_name"])


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
            resolution = resolve_layout(session, settings.model_root, image)
            slots = resolution.slots
        else:
            # A layout the operator has just confirmed by hand.
            resolution = LayoutResolution(
                slots=corrected_slots,
                state=USER_VERIFIED,
                source="user_verified",
                confidence=1.0,
                fingerprint=camera_signature(image).fingerprint,
                message="Layout verified by the operator.",
                detector_available=True,
            )
            slots = corrected_slots
        layout = DetectedLayout(
            media_asset_id=media.id,
            model_name=resolution.source,
            layout_identifier=uuid.uuid4().hex,
            confidence=float(resolution.confidence),
            status="corrected" if corrected_slots is not None else "automatic",
            verification_state=resolution.state,
            verified_layout_id=resolution.verified_layout_id,
            camera_fingerprint=resolution.fingerprint,
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
        if not resolution.usable:
            # Geometry could not be established well enough to report occupancy,
            # and none is asserted.  Vehicle detection is independent of the bay
            # map, though, so the scene is still described: "I cannot read this
            # car park, but here is what is parked in it" is a far more useful
            # and more honest answer than an empty result.  The proposals are
            # saved so the advanced correction path can start from them.
            # Only the vehicle pass is run here; the layout has already been
            # resolved above and re-entering the full product path would repeat
            # that detection for nothing.  With no bays there is no parking area
            # either, so every vehicle's position is reported as unknown rather
            # than guessed at.
            vehicles, vehicles_available, diagnostics = detect_vehicles_with_diagnostics(
                image, settings
            )
            unresolved = SceneAnalysis(
                mode=PRODUCT_MODE,
                slots=[],
                vehicles=vehicles,
                vehicle_detection_available=vehicles_available,
                placements=[AREA_UNKNOWN] * len(vehicles),
            )
            # The scene is still rendered.  Suppressing the picture because the
            # bay map failed would hide the vehicle detection that did succeed,
            # and leave the reader unable to tell "nothing is here" from "I
            # could not read the bays".  Candidate geometry is drawn in a
            # neutral colour so it reads as a proposal, never as a verdict, and
            # the image is attached to the layout rather than to an analysis
            # record so no zero-space run reaches history or analytics.
            relative_unresolved = (
                Path("media") / "results" / "images" / f"{uuid.uuid4().hex}.jpg"
            )
            _overlay(
                image,
                [],
                settings.parking_data_root / relative_unresolved,
                vehicles=[vehicle.as_dict() for vehicle in vehicles],
                candidates=slots,
            )
            layout.preview_image_path = relative_unresolved.as_posix()
            session.commit()
            return {
                **resolution.as_dict(),
                **unresolved.as_dict(),
                "media_asset_id": media.id,
                "layout_id": layout.id,
                "localization_confidence": round(float(resolution.confidence), 6),
                "advanced_correction_available": True,
                # Reported for an abstention too: how long the system took to
                # decide it could not read the scene is a real operational
                # figure, and leaving it out made the UI show a blank where
                # every other run shows a number.
                "processing_time_ms": round((perf_counter() - started) * 1_000, 3),
                "candidate_spaces": len(slots),
                "candidate_polygons": [slot["polygon"] for slot in slots],
                "result_image_url": f"/api/v1/media/layouts/{layout.id}/image",
                # Why the vehicle count is what it is.  Without this a zero
                # reads as "the car park is empty" when it can equally mean
                # "this viewpoint defeats the detector".
                "scene_diagnostics": diagnostics.as_dict() if diagnostics else None,
            }
        # Product mode: the whole visible scene, not only the bays that were
        # proposed.  Vehicle evidence is fused with the crop classifier, so a
        # clearly visible vehicle is no longer invisible to the system merely
        # because its bay was missed.
        scene = analyse_product(session, settings, image, slots=slots)
        predictions = [slot.as_dict() for slot in scene.slots]
        inference_ms = scene.inference_ms
        model_name = scene.model_name
        occupied, uncertain, vacant = scene.occupied, scene.uncertain, scene.vacant
        relative_result = Path("media") / "results" / "images" / f"{uuid.uuid4().hex}.jpg"
        result_path = settings.parking_data_root / relative_result
        _overlay(
            image,
            predictions,
            result_path,
            vehicles=[vehicle.as_dict() for vehicle in scene.vehicles],
            unmapped=set(scene.unmapped_vehicle_indexes),
        )
        record = AnalysisRecord(
            dataset="User upload",
            scenario_id=f"upload:{media.sha256[:20]}",
            total_spaces=len(predictions),
            occupied_spaces=occupied,
            vacant_spaces=vacant,
            processing_time_ms=(perf_counter() - started) * 1_000,
            result_image_path=relative_result.as_posix(),
            model_name=model_name,
            average_confidence=sum(float(value["confidence"]) for value in predictions)
            / len(predictions),
            prediction_json=json.dumps(predictions, separators=(",", ":")),
            source_type="image_upload",
            media_asset_id=media.id,
            layout_id=layout.id,
            localization_confidence=float(resolution.confidence),
            result_status=resolution.state,
        )
        session.add(record)
        session.commit()
        session.refresh(record)
    return {
        **resolution.as_dict(),
        "status": "success",
        "analysis_id": record.id,
        "media_asset_id": media.id,
        "layout_id": layout.id,
        "correction_of_layout_id": correction_of_layout_id,
        "total_spaces": record.total_spaces,
        "occupied_spaces": record.occupied_spaces,
        "vacant_spaces": record.vacant_spaces,
        "uncertain_spaces": uncertain,
        # Everything product mode knows about the scene, including where each
        # vehicle sits relative to the facility.
        **{
            key: value
            for key, value in scene.as_dict().items()
            if key not in {"predictions", "total_spaces", "occupied_spaces", "vacant_spaces"}
        },
        "average_confidence": round(float(record.average_confidence or 0), 6),
        "localization_confidence": round(float(record.localization_confidence or 0), 6),
        "processing_time_ms": round(record.processing_time_ms, 3),
        "inference_time_ms": round(inference_ms, 3),
        "predictions": predictions,
        "result_image_url": f"/api/v1/media/analyses/{record.id}/image",
    }
