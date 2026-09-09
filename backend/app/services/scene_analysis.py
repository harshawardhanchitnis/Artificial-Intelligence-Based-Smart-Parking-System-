"""Understand a whole parking scene, or score a benchmark faithfully.

These are two different questions and the system was previously answering only
one of them while presenting it as the other.

**Benchmark mode** asks: given the bays this dataset defines, what is the
occupancy of each?  It uses exactly the supplied polygons, adds nothing the
detector happens to find, and runs the occupancy classifier alone, so a figure
produced today is comparable with one produced before any of this existed.

**Product mode** asks: what is actually in front of this camera?  It establishes
its own geometry, detects vehicles across the entire frame, associates the two
and fuses the evidence.  It may legitimately report more bays than a benchmark
defines, and more vehicles than there are bays.

Keeping them apart matters for honesty in both directions.  A benchmark image
routinely contains vehicles outside the labelled region -- a coach on the road
behind, cars in an aisle -- and product mode should see them.  But finding them
is not evidence about the benchmark, and a product-mode count must never be
reported as agreement with ground truth the dataset does not contain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.ml.association import AssociationResult, associate
from app.ml.geometry import rectify_slots
from app.ml.line_refinement import refine_all
from app.ml.occupancy_fusion import FusionPolicy, OccupancyEvidence, load_policy
from app.ml.occupancy_v3 import OccupancyV3Predictor
from app.ml.parking_area import (
    AREA_UNKNOWN,
    IN_PARKING_AREA,
    OUTSIDE_PARKING_AREA,
    ParkingArea,
    classify_vehicle_positions,
    infer_parking_area,
    summarise,
)
from app.ml.vehicle_detector import (
    SceneDiagnostics,
    VehicleDetection,
    VehicleDetector,
    VehicleDetectorNotReadyError,
    unclassified_count,
    vehicle_counts,
)
from app.services.localization_service import LayoutResolution, resolve_layout

# The two modes a result can be produced in.  Benchmark mode itself lives in
# ``app.ml.inference.analyse_scenario`` -- it predates this module and must
# keep producing byte-identical figures, so it was left where it is rather
# than reimplemented here.  The constant is the contract both paths label
# their responses with.
BENCHMARK_MODE = "benchmark"
PRODUCT_MODE = "product"


@dataclass
class SceneSlot:
    """One bay, with the evidence behind its verdict."""

    index: int
    polygon: list[list[float]]
    state: str
    occupied_probability: float
    confidence: float
    classifier_probability: float
    localization_confidence: float = 1.0
    vehicle_class: str | None = None
    vehicle_indexes: list[int] = field(default_factory=list)
    geometry_refined: bool = False
    evidence: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "polygon": self.polygon,
            "occupancy_state": self.state,
            # Kept for every existing consumer: an uncertain bay is not occupied.
            "predicted_occupied": self.state == "occupied",
            "occupied_probability": round(self.occupied_probability, 6),
            "confidence": round(self.confidence, 6),
            "classifier_probability": round(self.classifier_probability, 6),
            "localization_confidence": round(self.localization_confidence, 6),
            "corner_confidence": round(self.localization_confidence, 6),
            "vehicle_class": self.vehicle_class,
            "vehicle_indexes": self.vehicle_indexes,
            "geometry_refined": self.geometry_refined,
            "evidence": {key: round(value, 4) for key, value in self.evidence.items()},
        }


@dataclass
class SceneAnalysis:
    mode: str
    slots: list[SceneSlot]
    vehicles: list[VehicleDetection] = field(default_factory=list)
    unmapped_vehicle_indexes: list[int] = field(default_factory=list)
    layout: LayoutResolution | None = None
    vehicle_detection_available: bool = False
    fusion_applied: bool = False
    inference_ms: float = 0.0
    model_name: str = ""
    area: ParkingArea = field(default_factory=ParkingArea)
    placements: list[str] = field(default_factory=list)

    @property
    def facility_vehicles(self) -> list[VehicleDetection]:
        """Vehicles on the site, excluding traffic that merely passes it.

        Every operational figure is derived from this list rather than from
        every detection in the frame, so a busy road behind a car park cannot
        inflate the site's numbers.
        """
        if not self.placements:
            return self.vehicles
        return [
            vehicle
            for vehicle, placement in zip(self.vehicles, self.placements, strict=True)
            if placement != OUTSIDE_PARKING_AREA
        ]

    @property
    def occupied(self) -> int:
        return sum(1 for slot in self.slots if slot.state == "occupied")

    @property
    def uncertain(self) -> int:
        return sum(1 for slot in self.slots if slot.state == "uncertain")

    @property
    def vacant(self) -> int:
        return len(self.slots) - self.occupied - self.uncertain

    def as_dict(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "analysis_mode": self.mode,
            "total_spaces": len(self.slots),
            "occupied_spaces": self.occupied,
            "vacant_spaces": self.vacant,
            "uncertain_spaces": self.uncertain,
            "predictions": [slot.as_dict() for slot in self.slots],
            "vehicle_detection_available": self.vehicle_detection_available,
            "fusion_applied": self.fusion_applied,
        }
        if self.mode == PRODUCT_MODE:
            facility = self.facility_vehicles
            payload["vehicles"] = [vehicle.as_dict() for vehicle in self.vehicles]
            # Class counts describe the facility, not the frame: a car on the
            # road behind the site is not one of the site's vehicles.
            payload["vehicle_counts"] = vehicle_counts(facility)
            payload["unclassified_objects"] = unclassified_count(facility)
            payload["scene_vehicle_counts"] = vehicle_counts(self.vehicles)
            payload["unmapped_vehicles"] = sum(
                1 for placement in self.placements if placement == IN_PARKING_AREA
            )
            payload["vehicles_outside_parking_area"] = sum(
                1 for placement in self.placements if placement == OUTSIDE_PARKING_AREA
            )
            payload["vehicle_placements"] = summarise(self.placements)
            payload["vehicle_placement_labels"] = self.placements
            payload.update(self.area.as_dict())
        return payload


def detect_vehicles(
    image: Image.Image, settings: Settings
) -> tuple[list[VehicleDetection], bool]:
    """Full-frame vehicle detection, or an honest empty result.

    A missing detector is not an error.  It degrades the system to bay-only
    occupancy, which is what it did before, so the caller is told the evidence
    was unavailable rather than being handed an empty list that reads as "no
    vehicles are present".
    """
    return detect_vehicles_with_diagnostics(image, settings)[:2]


def detect_vehicles_with_diagnostics(
    image: Image.Image, settings: Settings
) -> tuple[list[VehicleDetection], bool, SceneDiagnostics | None]:
    """As ``detect_vehicles``, also reporting what the detector rejected."""
    if not settings.enable_vehicle_detection:
        return [], False, None
    try:
        detector = VehicleDetector.load(settings.model_root)
    except VehicleDetectorNotReadyError:
        return [], False, None
    detections, diagnostics = detector.detect_with_diagnostics(image)
    return detections, True, diagnostics


def _classifier_states(
    image: Image.Image, polygons: list[list[list[float]]], model_root: Path
) -> tuple[list[dict[str, object]], float, str]:
    classifier = OccupancyV3Predictor.load(model_root)
    patches = rectify_slots(image, polygons)
    states, elapsed = classifier.predict(patches)
    return states, elapsed, str(classifier.metadata["model_name"])


def _fuse(
    policy: FusionPolicy,
    states: list[dict[str, object]],
    slots: list[dict[str, object]],
    polygons: list[list[list[float]]],
    association: AssociationResult,
    vehicles: list[VehicleDetection],
    vehicle_available: bool,
    refined_flags: list[bool],
) -> list[SceneSlot]:
    scene_slots: list[SceneSlot] = []
    for index, (slot, polygon, state) in enumerate(
        zip(slots, polygons, states, strict=True)
    ):
        links = association.links_for_slot(index)
        occupying = [link for link in links if link.occupies]
        best = max(occupying, key=lambda link: link.slot_coverage, default=None)
        vehicle = vehicles[best.vehicle_index] if best is not None else None
        evidence = OccupancyEvidence(
            occupancy_probability=float(state["occupied_probability"]),
            geometry_confidence=float(slot.get("localization_confidence", 1.0)),
            vehicle_evidence_available=vehicle_available,
            vehicle_detected=vehicle is not None,
            vehicle_confidence=vehicle.confidence if vehicle is not None else 0.0,
            slot_coverage=best.slot_coverage if best is not None else 0.0,
            vehicle_coverage=best.vehicle_coverage if best is not None else 0.0,
        )
        fused = policy.fuse(evidence)
        scene_slots.append(
            SceneSlot(
                index=index + 1,
                polygon=polygon,
                state=fused.state,
                occupied_probability=fused.probability,
                confidence=fused.confidence,
                classifier_probability=float(state["occupied_probability"]),
                localization_confidence=float(slot.get("localization_confidence", 1.0)),
                vehicle_class=vehicle.product_class if vehicle is not None else None,
                vehicle_indexes=sorted({link.vehicle_index for link in occupying}),
                geometry_refined=refined_flags[index],
                evidence=fused.contributions,
            )
        )
    return scene_slots


def analyse_product(
    session: Session,
    settings: Settings,
    image: Image.Image,
    *,
    slots: list[dict[str, object]] | None = None,
) -> SceneAnalysis:
    """Understand the whole visible scene and decide occupancy from all of it.

    ``slots`` short-circuits layout resolution for a camera whose geometry is
    already established -- a verified layout, or one a video job calibrated
    across several frames -- so the expensive detection is not repeated.
    """
    layout: LayoutResolution | None = None
    if slots is None:
        layout = resolve_layout(session, settings.model_root, image)
        slots = layout.slots
        if not layout.usable:
            # Geometry is still never guessed at.  But vehicle detection does
            # not depend on the bay map, so the scene is reported anyway: every
            # vehicle is unmapped, which is the honest description of a car park
            # whose layout could not be established, and it is far more useful
            # than an empty result.  It is also the signal that distinguishes
            # "nothing is here" from "I could not read the bays".
            vehicles, available = detect_vehicles(image, settings)
            # With no bays there is no parking area either, so every vehicle is
            # reported as a scene vehicle whose position cannot be judged.
            return SceneAnalysis(
                mode=PRODUCT_MODE,
                slots=[],
                vehicles=vehicles,
                unmapped_vehicle_indexes=list(range(len(vehicles))),
                layout=layout,
                vehicle_detection_available=available,
                placements=classify_vehicle_positions(ParkingArea(), [], set()) or
                [AREA_UNKNOWN] * len(vehicles),
            )

    polygons: list[list[list[float]]] = [slot["polygon"] for slot in slots]  # type: ignore[misc]
    refined_flags = [False] * len(polygons)
    if settings.enable_marking_refinement and polygons:
        refinements = refine_all(np.asarray(image.convert("RGB")), polygons)
        polygons = [refinement.polygon for refinement in refinements]
        refined_flags = [refinement.refined for refinement in refinements]

    vehicles, vehicle_available = detect_vehicles(image, settings)
    width, height = image.size
    association = associate(
        [[[x * width, y * height] for x, y in polygon] for polygon in polygons],
        [
            (
                vehicle.box[0] * width,
                vehicle.box[1] * height,
                vehicle.box[2] * width,
                vehicle.box[3] * height,
            )
            for vehicle in vehicles
        ],
    )

    states, elapsed, model_name = _classifier_states(image, polygons, settings.model_root)
    policy = load_policy(settings.model_root)
    scene_slots = _fuse(
        policy,
        states,
        slots,
        polygons,
        association,
        vehicles,
        vehicle_available,
        refined_flags,
    )

    # The facility's extent, derived from the bays and the vehicles standing on
    # them, so passing traffic can be excluded from operational figures.
    area = infer_parking_area(polygons, [vehicle.box for vehicle in vehicles])
    mapped = {
        link.vehicle_index
        for index in range(len(polygons))
        for link in association.occupying_links(index)
    }
    placements = classify_vehicle_positions(
        area, [vehicle.box for vehicle in vehicles], mapped
    )

    return SceneAnalysis(
        mode=PRODUCT_MODE,
        slots=scene_slots,
        vehicles=vehicles,
        unmapped_vehicle_indexes=association.unmapped_vehicles,
        layout=layout,
        vehicle_detection_available=vehicle_available,
        fusion_applied=vehicle_available and policy.fitted,
        inference_ms=elapsed,
        model_name=model_name,
        area=area,
        placements=placements,
    )
