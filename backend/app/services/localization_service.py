"""Resolve the parking-space geometry for an image, or decline to.

Three sources can supply geometry, in decreasing order of trust:

1. a layout already established for this camera, by calibration or by a person,
2. a registered benchmark camera whose canonical polygons are known,
3. the generalized detector, for a lot nobody has seen before.

The V2 pipeline had only source 2 and no way to decline, so an unfamiliar lot
was answered with a registered camera's polygons at a plausible-looking
confidence.  Two things prevent that here.  A recalled layout is *checked
against independently detected geometry* before it is trusted -- a classifier
score alone was measured to be non-discriminative, so it is not relied on.  And
when nothing can be established the result is ``automatic_layout_unresolved``,
never a confident answer.

That last state is deliberately not a request for help.  The product is meant
to run without a human in the loop, so the failure mode has to be the system
saying it cannot read this car park, not the system asking someone to draw it.
Callers are still handed the candidate geometry, and full-scene vehicle
detection runs regardless, so an unresolved layout still yields a useful and
honest answer: what is visibly parked here, and that the bays could not be
mapped.  Manual correction remains available as an explicit advanced action for
anyone who wants it, but nothing in the normal path depends on it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import VerifiedLayout
from app.ml.camera_fingerprint import (
    CameraSignature,
    camera_signature,
    decode_structure,
    encode_structure,
    same_camera,
)
from app.ml.generalized_localizer import DetectorNotReadyError, GeneralizedDetector
from app.ml.geometry import polygon_iou
from app.ml.localization import LocalizerNotReadyError, SlotLocalizerPredictor

# Share of independent detections a recalled layout must explain before it is
# trusted.  Correct layouts measured 0.69-0.89 and wrong ones 0.02-0.08, so this
# sits with roughly a two-fold margin on both sides.
LAYOUT_AGREEMENT_THRESHOLD = 0.35

# Fewer detections than this makes the ratio meaningless, so corroboration is
# treated as unavailable rather than passing on one or two boxes.
MIN_CORROBORATING_DETECTIONS = 5

# IoU at which a detected space is taken to corroborate a recalled one.
CORROBORATION_IOU = 0.30

# Below this mean detection confidence the proposal goes to verification rather
# than being reported as an automatic result.
AUTOMATIC_CONFIDENCE = 0.55

# A parking view with fewer spaces than this is more likely a false detection
# than a car park.
MIN_PLAUSIBLE_SLOTS = 4

# States a layout can be in, in the order the interface presents them.
AUTO_DETECTED = "auto_detected"
VERIFICATION_REQUIRED = "verification_required"
USER_VERIFIED = "user_verified"
REUSED_VERIFIED = "reused_verified"

# The AI's own verdict that it cannot read this car park.  Distinct from
# ``VERIFICATION_REQUIRED``, which asks a person to intervene: this state asks
# for nothing and asserts nothing.  A product that is meant to run without a
# human in the loop needs a way to fail that does not summon one, and declining
# is a legitimate answer where inventing geometry is not.
AUTOMATIC_LAYOUT_UNRESOLVED = "automatic_layout_unresolved"

# Nothing in the frame looked like a car park at all -- too few candidate spaces
# to be a parking view rather than a layout that could not be pinned down.  Kept
# separate because the two mean different things to a user: one says "I cannot
# read this car park", the other says "I do not think this is one".
INSUFFICIENT_VISUAL_EVIDENCE = "insufficient_visual_evidence"

# The vocabulary the product speaks, and the internal state each one comes from.
#
# The internal names are kept because they are written to the database and to
# stored records, but nothing user-facing should use them: a person needs to
# know whether the system has a layout, is working one out, or has given up --
# not which of four code paths produced it.
DETECTING_LAYOUT = "DETECTING_LAYOUT"
CALIBRATING = "CALIBRATING"
LAYOUT_ESTABLISHED = "LAYOUT_ESTABLISHED"
RECALIBRATING = "RECALIBRATING"
LAYOUT_UNRESOLVED = "AUTOMATIC_LAYOUT_UNRESOLVED"
NO_VISUAL_EVIDENCE = "INSUFFICIENT_VISUAL_EVIDENCE"

PRODUCT_STATES: dict[str, str] = {
    AUTO_DETECTED: LAYOUT_ESTABLISHED,
    REUSED_VERIFIED: LAYOUT_ESTABLISHED,
    USER_VERIFIED: LAYOUT_ESTABLISHED,
    VERIFICATION_REQUIRED: LAYOUT_UNRESOLVED,
    AUTOMATIC_LAYOUT_UNRESOLVED: LAYOUT_UNRESOLVED,
    INSUFFICIENT_VISUAL_EVIDENCE: NO_VISUAL_EVIDENCE,
}


def product_state(state: str) -> str:
    """The user-facing name for an internal layout state."""
    return PRODUCT_STATES.get(state, LAYOUT_UNRESOLVED)


@dataclass
class LayoutResolution:
    slots: list[dict[str, object]] = field(default_factory=list)
    state: str = VERIFICATION_REQUIRED
    source: str = "none"
    confidence: float = 0.0
    fingerprint: str = ""
    verified_layout_id: int | None = None
    message: str = ""
    detector_available: bool = False

    @property
    def usable(self) -> bool:
        """Whether occupancy may be reported without human verification."""
        return self.state in {AUTO_DETECTED, USER_VERIFIED, REUSED_VERIFIED} and bool(self.slots)

    def as_dict(self) -> dict[str, object]:
        return {
            "status": "success" if self.usable else self.state,
            "verification_state": self.state,
            "product_state": product_state(self.state),
            "geometry_source": self.source,
            "confidence": round(self.confidence, 6),
            "camera_fingerprint": self.fingerprint,
            "verified_layout_id": self.verified_layout_id,
            "detected_spaces": len(self.slots),
            "message": self.message,
            "detector_available": self.detector_available,
        }


def layout_agreement(recalled: list[dict[str, object]], detected: list[dict[str, object]]) -> float:
    """Share of *detections* that the recalled layout explains.

    This is the check that stops a recalled layout being stamped onto an image
    it does not belong to.  It deliberately asks "does this layout account for
    what the detector sees?" rather than "did the detector find every recalled
    space?": detector recall varies widely between cameras, so the second
    question rejects correct layouts on cameras the detector happens to find
    hard, while the first stays reliable.

    Measured on the audit's own cases -- correct layouts score 0.69 to 0.89,
    layouts belonging to a different car park score 0.02 to 0.08.
    """
    if not recalled or len(detected) < MIN_CORROBORATING_DETECTIONS:
        return 0.0
    explained = 0
    for candidate in detected:
        polygon = candidate["polygon"]
        if any(
            polygon_iou(polygon, slot["polygon"]) >= CORROBORATION_IOU  # type: ignore[arg-type]
            for slot in recalled
        ):
            explained += 1
    return explained / len(detected)


def find_verified_layout(session: Session, signature: CameraSignature) -> VerifiedLayout | None:
    """Look up a layout a person previously verified for this camera."""
    for layout in session.scalars(select(VerifiedLayout)).all():
        stored = decode_structure(
            layout.structure_json,
            layout.image_width,
            layout.image_height,
            layout.camera_fingerprint,
        )
        if same_camera(
            signature,
            layout.camera_fingerprint,
            layout.image_width,
            layout.image_height,
            stored_signature=stored,
        ):
            return layout
    return None


def _mean_confidence(slots: list[dict[str, object]]) -> float:
    if not slots:
        return 0.0
    values = [float(slot.get("localization_confidence", 0.0)) for slot in slots]
    return sum(values) / len(values)


def resolve_layout(session: Session, model_root: Path, image: Image.Image) -> LayoutResolution:
    """Establish the parking-space geometry for one image."""
    signature = camera_signature(image)
    resolution = LayoutResolution(fingerprint=signature.fingerprint)

    # 1 - a layout a person already verified for this camera.
    verified = find_verified_layout(session, signature)
    if verified is not None:
        verified.times_reused += 1
        verified.last_used_at = datetime.now(UTC)
        session.commit()
        return LayoutResolution(
            slots=json.loads(verified.geometry_json),
            state=REUSED_VERIFIED,
            source="verified_layout",
            confidence=1.0,
            fingerprint=signature.fingerprint,
            verified_layout_id=verified.id,
            message=f"Reusing the layout you verified for {verified.display_name}.",
            detector_available=True,
        )

    detected: list[dict[str, object]] = []
    detector_available = False
    try:
        detector = GeneralizedDetector.load(model_root)
        detected = detector.detect(image)
        detector_available = True
    except DetectorNotReadyError:
        detected = []
    resolution.detector_available = detector_available

    # 2 - a registered benchmark camera, but only if detection corroborates it.
    recalled: list[dict[str, object]] = []
    recalled_confidence = 0.0
    try:
        registered = SlotLocalizerPredictor.load(model_root).detect(image)
        if registered["status"] != "unsupported_layout":
            recalled = list(registered["slots"])  # type: ignore[arg-type]
            recalled_confidence = float(registered["confidence"])
    except (LocalizerNotReadyError, OSError, ValueError):
        recalled = []

    if recalled:
        if not detector_available:
            # Without an independent check the recalled layout cannot be
            # corroborated, so it is offered for verification rather than
            # reported as established fact.
            return LayoutResolution(
                slots=recalled,
                state=VERIFICATION_REQUIRED,
                source="registered_camera",
                confidence=recalled_confidence,
                fingerprint=signature.fingerprint,
                message=(
                    "A registered camera layout matched, but the detector that "
                    "corroborates it is unavailable. Confirm the spaces before use."
                ),
                detector_available=False,
            )
        agreement = layout_agreement(recalled, detected)
        if agreement >= LAYOUT_AGREEMENT_THRESHOLD:
            return LayoutResolution(
                slots=recalled,
                state=AUTO_DETECTED,
                source="registered_camera",
                confidence=recalled_confidence,
                fingerprint=signature.fingerprint,
                message=(
                    f"Recognised a registered camera; independent detection "
                    f"corroborated {agreement:.0%} of what it found."
                ),
                detector_available=True,
            )

    # 3 - generalized detection for a lot nobody has registered.
    if len(detected) >= MIN_PLAUSIBLE_SLOTS:
        confidence = _mean_confidence(detected)
        automatic = confidence >= AUTOMATIC_CONFIDENCE
        return LayoutResolution(
            slots=detected,
            state=AUTO_DETECTED if automatic else AUTOMATIC_LAYOUT_UNRESOLVED,
            source="generalized_detector",
            confidence=confidence,
            fingerprint=signature.fingerprint,
            message=(
                f"Detected {len(detected)} parking spaces in a previously unseen view."
                if automatic
                else (
                    "The parking layout could not be determined reliably from this "
                    f"image. {len(detected)} candidate spaces were found, but not "
                    "with enough confidence to report occupancy against them. A "
                    "fixed-camera clip of the same view can calibrate itself across "
                    "several frames, which usually succeeds where one image does not."
                )
            ),
            detector_available=True,
        )

    # Nothing that looked like a car park, as against a car park that could not
    # be pinned down.  The two get different states because they need different
    # answers from the user: one is worth retrying with a better view of the same
    # site, the other means this is not a parking scene.
    resolution.state = (
        INSUFFICIENT_VISUAL_EVIDENCE if not detected else AUTOMATIC_LAYOUT_UNRESOLVED
    )
    resolution.source = "generalized_detector" if detector_available else "none"
    resolution.slots = detected
    resolution.message = (
        "No parking spaces could be identified in this scene. Any vehicles found "
        "are still reported below."
        if not detected
        else (
            "The parking layout could not be determined reliably from this scene. "
            "Any vehicles found are still reported below. A fixed-camera clip of "
            "the same view can calibrate itself across several frames."
        )
    )
    return resolution


def save_verified_layout(
    session: Session,
    *,
    fingerprint: str,
    display_name: str,
    slots: list[dict[str, object]],
    width: int,
    height: int,
    signature: CameraSignature | None = None,
) -> VerifiedLayout:
    """Store a human-confirmed layout so this camera is recognised next time."""
    existing = session.scalar(
        select(VerifiedLayout).where(VerifiedLayout.camera_fingerprint == fingerprint)
    )
    geometry = json.dumps(slots, separators=(",", ":"))
    structure = encode_structure(signature) if signature is not None else None
    if existing is not None:
        existing.geometry_json = geometry
        existing.slot_count = len(slots)
        existing.display_name = display_name
        existing.image_width = width
        existing.image_height = height
        if structure is not None:
            existing.structure_json = structure
        session.commit()
        return existing
    layout = VerifiedLayout(
        camera_fingerprint=fingerprint,
        display_name=display_name,
        source="user_verified",
        slot_count=len(slots),
        geometry_json=geometry,
        image_width=width,
        image_height=height,
        structure_json=structure,
    )
    session.add(layout)
    session.commit()
    session.refresh(layout)
    return layout
