"""Associate detected vehicles with parking-space polygons.

Intersection-over-union is the wrong measure here and gets the two most common
cases backwards.  A lorry straddling two bays overlaps each of them at roughly
0.5 IoU, and a scooter parked in a full-size bay overlaps it at roughly 0.15 --
yet both bays are occupied.  Meanwhile a car in the aisle can clip the corner of
a bay at 0.3 IoU while occupying none of it.  A single symmetric ratio cannot
separate those, because the shapes being compared differ in both scale and
meaning: one is a region of ground, the other is an object standing on it.

So the two directions are kept apart:

``slot_coverage``     how much of the bay the vehicle covers -- high when a
                      vehicle fills or straddles the bay
``vehicle_coverage``  how much of the vehicle sits inside the bay -- high when a
                      small vehicle is properly parked, low when a large vehicle
                      merely overlaps the edge

Either one being high is evidence of occupancy; both being low is not.  That
asymmetry is what lets one lorry occupy two bays and one scooter occupy one.

A second correction is perspective.  A detector's box encloses the whole
vehicle, including the body standing above the ground, so it reaches past the
footprint on the side away from the camera.  Bay polygons are drawn on the
ground.  Comparing the two directly biases every association toward whichever
bay lies beyond the vehicle, so only the lower band of the box -- the part that
meets the ground -- is used for the comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from app.ml.geometry import order_polygon

# Overlap at which a vehicle is taken to occupy a bay, applied to whichever of
# the two coverages is the larger.
#
# Taking the maximum is intersection-over-*minimum*, and it is what makes the
# rule work in both directions at once.  A lorry straddling two bays covers
# roughly half of each, so the bay-side ratio carries it.  A scooter covers a
# twentieth of its bay but lies wholly inside it, so the vehicle-side ratio
# carries that.  Intersection-over-union would fail both, because it divides by
# a total that includes the part of the larger shape nobody is asking about.
OCCUPANCY_OVERLAP = 0.35

# Weakest evidence still recorded as a link, used to spot vehicles that overlap
# a bay without occupying it -- aisle parking and badly parked vehicles.
MIN_LINK_COVERAGE = 0.10

# Fraction of a vehicle box, measured up from its bottom edge, taken as the
# ground footprint.  The full box includes the vehicle's height, which projects
# away from the camera and would otherwise drag associations into the next bay.
GROUND_BAND = 0.45


@dataclass
class VehicleLink:
    """One vehicle's relationship to one bay."""

    vehicle_index: int
    slot_index: int
    slot_coverage: float
    vehicle_coverage: float
    centroid_inside: bool

    @property
    def occupies(self) -> bool:
        """Whether this link is strong enough to mean the bay is filled."""
        return max(self.slot_coverage, self.vehicle_coverage) >= OCCUPANCY_OVERLAP

    def as_dict(self) -> dict[str, object]:
        return {
            "vehicle_index": self.vehicle_index,
            "slot_index": self.slot_index,
            "slot_coverage": round(self.slot_coverage, 4),
            "vehicle_coverage": round(self.vehicle_coverage, 4),
            "centroid_inside": self.centroid_inside,
            "occupies": self.occupies,
        }


@dataclass
class AssociationResult:
    links: list[VehicleLink] = field(default_factory=list)
    unmapped_vehicles: list[int] = field(default_factory=list)

    def links_for_slot(self, slot_index: int) -> list[VehicleLink]:
        return [link for link in self.links if link.slot_index == slot_index]

    def occupying_links(self, slot_index: int) -> list[VehicleLink]:
        return [link for link in self.links_for_slot(slot_index) if link.occupies]


def ground_footprint(box: tuple[float, float, float, float]) -> np.ndarray:
    """The lower band of a vehicle box, as a polygon on the ground plane.

    An approximation rather than a projection: without camera calibration the
    true footprint is unknowable, but the bottom of the box is where the vehicle
    meets the ground in every view, so the band is a far better stand-in than
    the whole box.
    """
    x1, y1, x2, y2 = box
    top = y2 - (y2 - y1) * GROUND_BAND
    return np.array([[x1, top], [x2, top], [x2, y2], [x1, y2]], dtype=np.float32)


def _as_contour(polygon: list[list[float]]) -> np.ndarray:
    points = np.asarray(order_polygon(polygon), dtype=np.float32)
    if cv2.contourArea(points) < 0:
        points = points[::-1].copy()
    return points


def coverage(first: np.ndarray, second: np.ndarray) -> tuple[float, float, float]:
    """Intersection area, and its share of each shape."""
    first_area = float(cv2.contourArea(first))
    second_area = float(cv2.contourArea(second))
    if first_area <= 0 or second_area <= 0:
        return 0.0, 0.0, 0.0
    intersection, _ = cv2.intersectConvexConvex(first, second)
    intersection = float(intersection)
    return intersection, intersection / first_area, intersection / second_area


def associate(
    slots: list[list[list[float]]],
    vehicles: list[tuple[float, float, float, float]],
) -> AssociationResult:
    """Link vehicles to bays, in the same pixel coordinate space.

    ``slots`` are bay polygons; ``vehicles`` are detector boxes as
    ``(x1, y1, x2, y2)``.  A vehicle may link to several bays -- that is how a
    lorry straddling two spaces fills both -- and a bay may hold several links,
    which is how two motorcycles sharing one bay are both accounted for.
    """
    slot_contours = [_as_contour(slot) for slot in slots]
    result = AssociationResult()

    for vehicle_index, box in enumerate(vehicles):
        footprint = ground_footprint(box)
        centre = footprint.mean(axis=0)
        linked = False
        for slot_index, slot in enumerate(slot_contours):
            _, slot_share, vehicle_share = coverage(slot, footprint)
            if max(slot_share, vehicle_share) < MIN_LINK_COVERAGE:
                continue
            inside = cv2.pointPolygonTest(slot, (float(centre[0]), float(centre[1])), False) >= 0
            link = VehicleLink(
                vehicle_index=vehicle_index,
                slot_index=slot_index,
                slot_coverage=slot_share,
                vehicle_coverage=vehicle_share,
                centroid_inside=bool(inside),
            )
            result.links.append(link)
            if link.occupies:
                linked = True
        if not linked:
            # Not necessarily illegal parking: it may equally mean the bay was
            # never detected.  The product reports it neutrally as unmapped.
            result.unmapped_vehicles.append(vehicle_index)

    return result
