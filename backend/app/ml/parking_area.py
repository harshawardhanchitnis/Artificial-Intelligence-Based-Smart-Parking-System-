"""Infer where the parking facility is, so passing traffic stays out of it.

Detecting vehicles across the whole frame is what makes the system notice cars
its bay map missed.  It also makes it notice cars that have nothing to do with
the car park: on one CNRPark view the detector found 72 vehicles against 35
mapped bays, and a large share of the difference was traffic on the road behind
the site.  Counting those as "unmapped vehicles" would inflate every operational
figure the product reports, and would do it worst exactly where the site is
busiest with through traffic.

The parking area is derived from the bays themselves rather than configured.
Bay polygons are painted into a mask and morphologically closed, which bridges
the aisles between rows -- an aisle is bounded by bays on both sides -- without
reaching across a kerb into a road, because a road has no bays on its far side
to close against.  The result follows the real shape of the site, including
L-shaped and split lots, and no rectangular region is ever assumed.

The closing radius is expressed in units of the local bay size rather than
pixels, so it scales with the camera: bays near the lens are large and their
aisles are wide, bays at the top of the frame are small and their aisles narrow,
and one setting covers both.

Where no layout could be established there is no parking area, and the module
says so rather than guessing.  Every vehicle is then simply a scene vehicle.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Working resolution for the mask.  The region is a coarse spatial concept and
# nothing downstream needs it sharper than this; keeping it small makes the
# morphology cheap enough to run on every frame.
MASK_SIZE = 384

# Closing radius as a multiple of the median bay's shorter side.  An aisle is
# typically one to two bay widths across, so this bridges rows while leaving a
# kerb-width gap to the road unbridged.
CLOSING_BAY_MULTIPLE = 1.6

# A region smaller than this share of the total bay area is a stray fragment.
MIN_REGION_AREA_SHARE = 0.05

# How far from bay-anchored surface a vehicle may sit and still be treated as
# evidence of parking surface, again in bay widths.
#
# This closes a circular argument that the previous version was open to. Vehicle
# footprints seeded the region unconditionally, so a false detection on a road
# could grow a region around itself, then be found "inside the parking area" by
# the very region it had created, and so corroborate itself. Requiring vehicle
# evidence to be anchored to geometry the bay detector independently found means
# a vehicle can *extend* the facility but never *invent* it.
#
# The value is deliberately generous. It has to keep the case this seeding
# exists for: CNRPark camera 1 labels two rows of a five-row car park, and the
# unlabelled rows are several bay widths from the labelled ones.
VEHICLE_SUPPORT_BAY_MULTIPLE = 6.0

# Where a vehicle sits relative to the parking area.
IN_MAPPED_SPACE = "in_mapped_space"
IN_PARKING_AREA = "in_parking_area"
OUTSIDE_PARKING_AREA = "outside_parking_area"
AREA_UNKNOWN = "area_unknown"


@dataclass
class ParkingArea:
    """The inferred extent of the parking facility, in normalised coordinates."""

    regions: list[np.ndarray] = field(default_factory=list)
    bay_scale: float = 0.0

    @property
    def known(self) -> bool:
        return bool(self.regions)

    def contains(self, x: float, y: float) -> bool:
        """Whether a normalised point lies inside the facility."""
        if not self.known:
            return False
        point = (float(x) * MASK_SIZE, float(y) * MASK_SIZE)
        return any(
            cv2.pointPolygonTest(region, point, False) >= 0 for region in self.regions
        )

    def outlines(self) -> list[list[list[float]]]:
        """The region boundaries, normalised, for overlays and diagnostics."""
        return [
            [[float(x) / MASK_SIZE, float(y) / MASK_SIZE] for x, y in region.reshape(-1, 2)]
            for region in self.regions
        ]

    def as_dict(self) -> dict[str, object]:
        return {
            "parking_area_known": self.known,
            "parking_area_regions": len(self.regions),
            "parking_area_outlines": self.outlines(),
        }


def _median_bay_side(polygons: list[list[list[float]]]) -> float:
    """Median of each bay's shorter side, in mask pixels."""
    sides: list[float] = []
    for polygon in polygons:
        points = np.asarray(polygon, dtype=np.float32) * MASK_SIZE
        if points.shape != (4, 2):
            continue
        edges = [
            float(np.linalg.norm(points[index] - points[(index + 1) % 4]))
            for index in range(4)
        ]
        # Opposite edges pair up; the shorter pair is the bay's width.
        sides.append(min((edges[0] + edges[2]) / 2, (edges[1] + edges[3]) / 2))
    return float(np.median(sides)) if sides else 0.0


def infer_parking_area(
    polygons: list[list[list[float]]],
    vehicle_boxes: list[tuple[float, float, float, float]] | None = None,
) -> ParkingArea:
    """Derive the facility's extent from the bays and the vehicles standing on it.

    Bays alone are not enough, because a benchmark rarely labels the whole site:
    CNRPark camera 1 annotates two rows of a car park that has five, so a region
    grown from its bays put forty genuinely parked cars "outside the facility".

    Parked vehicles are themselves evidence of parking surface, so they seed the
    mask too.  What keeps the road out is the closing radius rather than the
    seed: rows of parked cars sit within a bay-width of each other and merge,
    while traffic beyond a kerb is separated by more than that and stays its own
    fragment, which the area-share filter then discards.
    """
    if not polygons:
        return ParkingArea()

    mask = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.uint8)
    for polygon in polygons:
        points = (np.asarray(polygon, dtype=np.float32) * MASK_SIZE).astype(np.int32)
        if points.shape[0] >= 3:
            cv2.fillConvexPoly(mask, points, 1)
    if not mask.any():
        return ParkingArea()

    bay_side = _median_bay_side(polygons)

    # Vehicles may only extend surface the bay detector independently found.
    # Anything beyond that reach is ignored as a seed -- it can still be
    # detected, counted and drawn, it simply cannot vote itself a car park.
    support = max(
        3, int(round(max(bay_side, 1.0) * VEHICLE_SUPPORT_BAY_MULTIPLE))
    )
    anchor = cv2.dilate(
        mask,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (support * 2 + 1, support * 2 + 1)),
    )
    for box in vehicle_boxes or []:
        x1, y1, x2, y2 = (float(value) * MASK_SIZE for value in box)
        # Only the ground band of the vehicle, for the same reason the
        # association uses it: the box includes height that projects away from
        # the camera and would drag the region across a boundary.
        top = y2 - (y2 - y1) * 0.45
        left, right = int(x1), int(x2)
        upper, lower = int(top), int(y2)
        footprint = np.zeros_like(mask)
        cv2.rectangle(footprint, (left, upper), (right, lower), 1, -1)
        if not np.any(footprint & anchor):
            continue
        mask |= footprint
    radius = max(3, int(round(bay_side * CLOSING_BAY_MULTIPLE)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return ParkingArea()
    total = sum(float(cv2.contourArea(contour)) for contour in contours)
    regions = [
        contour
        for contour in contours
        if total > 0 and float(cv2.contourArea(contour)) / total >= MIN_REGION_AREA_SHARE
    ]
    return ParkingArea(regions=regions, bay_scale=bay_side / MASK_SIZE)


def classify_vehicle_positions(
    area: ParkingArea,
    vehicle_boxes: list[tuple[float, float, float, float]],
    mapped_indexes: set[int],
) -> list[str]:
    """Place every detected vehicle in one of the three product categories.

    Position is judged from the bottom edge of the box -- where the vehicle
    meets the ground -- rather than its centre, because a tall vehicle just
    outside the site would otherwise have its centre float over the car park
    behind it.
    """
    # The region is rasterised at MASK_SIZE, so a vehicle that helped define the
    # region's lower edge has its own bottom edge lying exactly on the boundary,
    # where the containment test can fall either side of it by one pixel. The
    # probe is lifted half a mask pixel to remove that discretisation artifact;
    # it is still the point where the vehicle meets the ground, not its centre.
    inset = 0.5 / MASK_SIZE
    placements: list[str] = []
    for index, (x1, _, x2, y2) in enumerate(vehicle_boxes):
        if index in mapped_indexes:
            placements.append(IN_MAPPED_SPACE)
        elif not area.known:
            placements.append(AREA_UNKNOWN)
        elif area.contains((x1 + x2) / 2.0, y2 - inset):
            placements.append(IN_PARKING_AREA)
        else:
            placements.append(OUTSIDE_PARKING_AREA)
    return placements


def summarise(placements: list[str]) -> dict[str, int]:
    """Counts per category, with every key present.

    Only the first two describe the facility.  ``outside_parking_area`` is
    reported so the figure is visible and auditable, and excluded from every
    operational metric.
    """
    return {
        IN_MAPPED_SPACE: placements.count(IN_MAPPED_SPACE),
        IN_PARKING_AREA: placements.count(IN_PARKING_AREA),
        OUTSIDE_PARKING_AREA: placements.count(OUTSIDE_PARKING_AREA),
        AREA_UNKNOWN: placements.count(AREA_UNKNOWN),
    }
