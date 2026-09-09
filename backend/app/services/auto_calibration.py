"""Establish a camera's parking layout from several frames, with nobody asked.

A single frame is a poor basis for a layout.  The detector's per-frame errors
are not stable -- a shadow that reads as a bay in one frame does not in the
next, and a bay half hidden by a passing van reappears once it has gone -- so a
layout built from one frame inherits every one of those accidents.

A fixed camera gives a way out that costs nothing: look at several frames and
keep only what recurs.  A real bay is in the same place in every frame because
it is painted on the ground.  A spurious detection is somewhere else, or gone,
by the next sample.  Agreement across frames is therefore direct evidence of
which proposals are real, and averaging the survivors is a free accuracy gain
because the per-frame corner noise is independent while the bay is not.

This is what replaces asking a person to draw the bays.  It is still allowed to
fail: a scene where nothing recurs produces ``unresolved`` and an honest message
rather than a layout assembled from whatever the last frame happened to contain.
Abstaining is a better product than confident invention.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime

import numpy as np
from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.db.models import VerifiedLayout
from app.ml.camera_fingerprint import CameraSignature, camera_signature, encode_structure
from app.ml.generalized_localizer import DetectorNotReadyError, GeneralizedDetector
from app.ml.geometry import order_polygon, polygon_iou

# Lifecycle a camera's layout moves through, without human involvement.
CALIBRATING = "calibrating"
ACTIVE = "active"
RECALIBRATING = "recalibrating"
UNRESOLVED = "unresolved"

# Overlap at which a proposal in a new frame is taken to be the same bay as one
# already tracked.  Deliberately loose: the point is to recognise the same bay
# despite per-frame corner jitter, and the consensus test that follows is what
# supplies the precision.
TRACK_IOU = 0.45

# A layout with fewer bays than this is not a car park.
MIN_LAYOUT_SLOTS = 4

# Share of a later frame's detections the established layout must still explain
# before the camera is considered unchanged.  Below it the view has moved and
# the layout is rebuilt rather than stamped onto footage it no longer describes.
#
# Measured with the shipped four-corner detector over two reconstructed
# sequences: frames from the camera a layout was built on score 0.200 to 1.000
# (median 0.636), and frames from a different car park score 0.000 to 0.100
# (median 0.017).  The threshold sits between those populations.
#
# Two honest caveats travel with it.  The margin is narrower than the
# corroboration threshold's -- 0.200 against 0.100 rather than 0.400 against
# 0.214 -- on a sample of eight frames per condition.  And a low score partly
# reflects layout *incompleteness* rather than movement: where calibration
# established only thirteen bays of a larger lot, detections elsewhere in the
# same lot go unexplained and drag the ratio down.  A previous value of 0.55 sat
# inside the same-camera population and raised drift alarms on a camera that had
# not moved.
DRIFT_AGREEMENT = 0.15


@dataclass
class BayTrack:
    """One candidate bay, accumulated across frames."""

    corners: list[np.ndarray] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    @property
    def seen(self) -> int:
        return len(self.corners)

    def mean_polygon(self) -> list[list[float]]:
        """Average the observations, which cancels independent corner noise."""
        stacked = np.mean(np.stack(self.corners), axis=0)
        return order_polygon([[float(x), float(y)] for x, y in stacked])

    def mean_confidence(self) -> float:
        return float(np.mean(self.confidences))


@dataclass
class CalibrationResult:
    state: str
    slots: list[dict[str, object]] = field(default_factory=list)
    frames_used: int = 0
    consensus: float = 0.0
    message: str = ""

    @property
    def established(self) -> bool:
        return self.state in {ACTIVE, RECALIBRATING} and len(self.slots) >= MIN_LAYOUT_SLOTS

    def as_dict(self) -> dict[str, object]:
        return {
            "calibration_state": self.state,
            "detected_spaces": len(self.slots),
            "frames_used": self.frames_used,
            "consensus": round(self.consensus, 4),
            "message": self.message,
        }


def calibrate_from_frames(
    frames: list[Image.Image], settings: Settings
) -> CalibrationResult:
    """Build a layout from several frames of one fixed camera.

    Returns ``unresolved`` rather than a guess whenever the frames do not agree
    on enough bays, which is the honest outcome for a view the detector cannot
    read.
    """
    if not frames:
        return CalibrationResult(UNRESOLVED, message="No frames were available to calibrate from.")

    try:
        detector = GeneralizedDetector.load(settings.model_root)
    except DetectorNotReadyError:
        return CalibrationResult(
            UNRESOLVED,
            message=(
                "The parking-space detector is not installed, "
                "so no layout could be established."
            ),
        )

    tracks: list[BayTrack] = []
    for frame in frames:
        detections = detector.detect(frame)
        claimed: set[int] = set()
        for detection in detections:
            polygon = detection["polygon"]
            confidence = float(detection.get("localization_confidence", 0.0))
            best_index, best_overlap = -1, 0.0
            for index, track in enumerate(tracks):
                if index in claimed:
                    continue
                overlap = polygon_iou(polygon, track.mean_polygon())  # type: ignore[arg-type]
                if overlap > best_overlap:
                    best_index, best_overlap = index, overlap
            if best_index >= 0 and best_overlap >= TRACK_IOU:
                tracks[best_index].corners.append(np.asarray(polygon, dtype=np.float64))
                tracks[best_index].confidences.append(confidence)
                claimed.add(best_index)
            else:
                track = BayTrack()
                track.corners.append(np.asarray(polygon, dtype=np.float64))
                track.confidences.append(confidence)
                tracks.append(track)

    frame_count = len(frames)
    required = max(2, round(settings.calibration_consensus * frame_count))
    survivors = [track for track in tracks if track.seen >= required]

    if len(survivors) < MIN_LAYOUT_SLOTS:
        return CalibrationResult(
            UNRESOLVED,
            frames_used=frame_count,
            consensus=0.0,
            message=(
                "The parking layout could not be determined reliably from this scene. "
                f"Only {len(survivors)} space(s) were found consistently across "
                f"{frame_count} frames."
            ),
        )

    slots = [
        {
            "polygon": track.mean_polygon(),
            "localization_confidence": round(track.mean_confidence(), 6),
            "corner_confidence": round(track.mean_confidence(), 6),
            "frames_seen": track.seen,
        }
        for track in sorted(survivors, key=lambda item: -item.seen)
    ]
    consensus = float(np.mean([track.seen / frame_count for track in survivors]))
    return CalibrationResult(
        state=ACTIVE,
        slots=slots,
        frames_used=frame_count,
        consensus=consensus,
        message=(
            f"Layout established automatically from {frame_count} frames; "
            f"{len(slots)} spaces appeared in {consensus:.0%} of them on average."
        ),
    )


def layout_still_describes(
    stored: list[dict[str, object]], frame: Image.Image, settings: Settings
) -> tuple[bool, float]:
    """Check a stored layout against a fresh frame, for drift detection.

    Compares against independently detected geometry rather than against a
    similarity score, because a camera nudged by wind or maintenance produces an
    image that still *looks* like the same scene while no longer matching it.
    """
    try:
        detector = GeneralizedDetector.load(settings.model_root)
    except DetectorNotReadyError:
        # No independent opinion is available, so nothing can be said about
        # drift.  Reporting "no drift" is the safe answer: it leaves the layout
        # in place rather than discarding a good one on absent evidence.
        return True, 0.0
    detections = detector.detect(frame)
    if not detections or not stored:
        return True, 0.0
    explained = sum(
        1
        for detection in detections
        if any(
            polygon_iou(detection["polygon"], slot["polygon"]) >= TRACK_IOU  # type: ignore[arg-type]
            for slot in stored
        )
    )
    agreement = explained / len(detections)
    return agreement >= DRIFT_AGREEMENT, agreement


def store_calibrated_layout(
    session: Session,
    *,
    signature: CameraSignature,
    display_name: str,
    result: CalibrationResult,
    width: int,
    height: int,
) -> VerifiedLayout:
    """Persist an automatically established layout for reuse by this camera."""
    existing = session.scalar(
        select(VerifiedLayout).where(
            VerifiedLayout.camera_fingerprint == signature.fingerprint
        )
    )
    geometry = json.dumps(result.slots, separators=(",", ":"))
    now = datetime.now(UTC)
    if existing is not None:
        existing.geometry_json = geometry
        existing.slot_count = len(result.slots)
        existing.image_width = width
        existing.image_height = height
        existing.lifecycle_state = ACTIVE
        existing.frames_observed = result.frames_used
        existing.consensus = result.consensus
        existing.revalidated_at = now
        existing.structure_json = encode_structure(signature)
        session.commit()
        return existing
    layout = VerifiedLayout(
        camera_fingerprint=signature.fingerprint,
        display_name=display_name,
        source="automatic_calibration",
        slot_count=len(result.slots),
        geometry_json=geometry,
        image_width=width,
        image_height=height,
        structure_json=encode_structure(signature),
        lifecycle_state=ACTIVE,
        frames_observed=result.frames_used,
        consensus=result.consensus,
        revalidated_at=now,
    )
    session.add(layout)
    session.commit()
    session.refresh(layout)
    return layout


def calibrate_and_store(
    session: Session,
    settings: Settings,
    frames: list[Image.Image],
    display_name: str,
) -> tuple[CalibrationResult, VerifiedLayout | None]:
    """Full onboarding for one camera: calibrate, then remember what was found."""
    result = calibrate_from_frames(frames, settings)
    if not result.established:
        return result, None
    reference = frames[0]
    signature = camera_signature(reference)
    layout = store_calibrated_layout(
        session,
        signature=signature,
        display_name=display_name,
        result=result,
        width=reference.width,
        height=reference.height,
    )
    return result, layout
