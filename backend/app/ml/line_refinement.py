"""Snap predicted bay corners onto the painted markings, where they exist.

A neural corner prediction is accurate to a few pixels at best, and the errors
it makes are not random: the detector was measured to find occupied bays at
0.711 recall and vacant ones at 0.433, because a parked vehicle is a far
stronger signal than a painted line.  So the network tends to outline *the
vehicle* rather than *the bay*, which is visible in the overlays as boxes that
sit slightly forward of the markings and inherit the vehicle's proportions.

The painted lines carry the information the network under-uses, and they are
strongest exactly where the network is weakest -- an empty bay is nothing but
its markings.  This module recovers them classically and pulls each edge of a
proposed quadrilateral onto the nearest supporting line segment.

Restraint is the hard part.  Faded paint, wet tarmac, shadows and kerb edges all
produce line segments, and a refinement that trusts them unconditionally will
confidently move a correct bay onto a crack in the ground.  Every snap therefore
has to clear a support threshold, and the refined quadrilateral is discarded
wholesale if it drifts too far from the proposal it started as.  Refusing to
refine is always an acceptable outcome; the proposal is returned unchanged and
the caller is told no evidence was found.

Measured limitation -- why this is disabled by default
------------------------------------------------------
The approach was evaluated by perturbing annotated bays by 0.10 of their own
scale and asking whether refinement pulled them back.  It did not.  Across 520
bays the mean overlap with the truth went from 0.7522 to 0.7510, and three
successive tightenings of the thresholds only reduced the harm rather than
turning it into a gain.

The reason is a property of the corpus rather than of the method, and it was
confirmed directly: sampling the marking response along annotated bay edges and
along ground displaced away from them gives a ratio of 2.27 on ACPDS but 0.91 on
PKLot.  ACPDS bays are drawn on the paint; PKLot bays are drawn around where
vehicles stand, so on PKLot the painted lines and the ground truth are different
things and moving toward the paint provably moves away from the truth.  PKLot is
86% of the evaluation bays, and refinement worsened 8 of the 8 PKLot bays it
touched while ACPDS came out level at 6 improved against 5 worsened.

Nor can the two be told apart at inference time, which is what would otherwise
have rescued the feature: PKLot car parks *are* painted, so scene marking
response does not separate them (ACPDS 0.293, PKLot 0.263, with overlapping
ranges).  The difference is in the annotation convention, which is not visible
in the image.

So the machinery is kept, guarded and tested, and left switched off behind
``settings.enable_marking_refinement``.  It should be reconsidered against a
corpus whose bay annotations follow the painted markings, where the premise test
above can be re-run with ``ml/measure_marking_premise.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

# Painted markings are thin bright ridges.  This is the structuring width used
# to separate them from the tarmac beneath; wider than a typical line in pixels
# so the whole stripe survives the top-hat.
PAINT_KERNEL = 9

# Hough parameters, deliberately permissive.  Precision comes from the per-edge
# support test below, not from finding only perfect lines here.
HOUGH_THRESHOLD = 40
HOUGH_MIN_LENGTH = 25
HOUGH_MAX_GAP = 8

# A segment supports an edge only if it is nearly parallel to it.
MAX_ANGLE_DELTA_DEGREES = 10.0

# ...and lies within this distance of it, as a fraction of the bay's own scale,
# so the tolerance shrinks for distant bays and grows for near ones.  Bays in a
# row sit roughly one scale apart, so a wide window reaches the *neighbouring*
# bay's stripe; measured at 0.35 the refinement was worse than no refinement.
MAX_OFFSET_RATIO = 0.14

# Minimum share of an edge's length that supporting segments must cover before
# the edge is moved.  This is what stops a short crack in the tarmac from
# repositioning a whole bay boundary.
MIN_EDGE_SUPPORT = 0.60

# Accepted segments are grouped by their perpendicular offset from the edge and
# only the strongest group is fitted.  Averaging across groups was measured to
# place the refined edge in the gap *between* two painted stripes, which is
# worse than leaving the proposal alone.
OFFSET_CLUSTER_RATIO = 0.05

# Ceiling on how far one edge may travel, as a fraction of the bay's scale.
# Refinement exists to correct a few pixels of drift.
MAX_SHIFT_RATIO = 0.16

# A bay's four edges are rarely all painted -- the two side stripes usually are,
# the head and tail often are not -- but moving on a single edge reshapes the
# quadrilateral from one line's evidence, which measured worse than leaving it
# alone.  Two independent edges is the minimum that constrains the correction.
MIN_SNAPPED_EDGES = 2

# The refined outline must sit on this much more marking response than the
# proposal, on a 0-1 scale, before it is accepted.
MIN_BRIGHTNESS_GAIN = 0.02

# Scene-level floor on marking response before any refinement is attempted.
# Only genuinely unmarked surfaces fall below it; see the measured limitation in
# the module docstring for why this floor cannot do more than that.
MIN_SCENE_RESPONSE = 0.10

# A refined quadrilateral is rejected if it overlaps its own proposal by less
# than this, or changes area by more than the tolerance below.  Refinement is
# meant to correct a few pixels of drift, not to relocate a bay.
MIN_REFINED_IOU = 0.70
MAX_AREA_RATIO = 1.45


@dataclass
class RefinedQuad:
    """One bay after refinement, with an honest account of what was used."""

    polygon: list[list[float]]
    refined: bool
    edges_snapped: int
    support: float
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "polygon": self.polygon,
            "geometry_refined": self.refined,
            "edges_snapped": self.edges_snapped,
            "marking_support": round(self.support, 4),
            "refinement_note": self.reason,
        }


@dataclass
class MarkingField:
    """The markings visible in one image, computed once and reused.

    Extraction is image-level rather than per-bay because a busy car park holds
    hundreds of bays and re-running the filter inside each one would repeat the
    same work over overlapping pixels.  The ridge response is kept alongside the
    segments so a candidate line can be checked for actual brightness: a
    Hough segment only says "something linear is here", and tarmac cracks,
    shadow boundaries and vehicle sills all answer to that description.
    """

    segments: np.ndarray = field(default_factory=lambda: np.empty((0, 4), dtype=np.float32))
    response: np.ndarray | None = None

    @property
    def empty(self) -> bool:
        return self.segments.shape[0] == 0

    def brightness_along(self, start: np.ndarray, end: np.ndarray, samples: int = 24) -> float:
        """Mean ridge response along a line, 0 when no response is available."""
        if self.response is None:
            return 0.0
        height, width = self.response.shape[:2]
        fractions = np.linspace(0.0, 1.0, samples)[:, None]
        points = start[None, :] + fractions * (end - start)[None, :]
        columns = np.clip(points[:, 0].astype(int), 0, width - 1)
        rows = np.clip(points[:, 1].astype(int), 0, height - 1)
        return float(np.mean(self.response[rows, columns])) / 255.0


def paint_response(image: np.ndarray) -> np.ndarray:
    """Isolate thin bright structures -- the painted markings.

    A white top-hat keeps features narrower than the structuring element and
    removes the slowly varying background, which is what separates a painted
    stripe from the tarmac, from a shadow edge and from a building.
    """
    grey = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (PAINT_KERNEL, PAINT_KERNEL))
    tophat = cv2.morphologyEx(grey, cv2.MORPH_TOPHAT, kernel)
    # Normalising per image keeps the threshold meaningful across a bright
    # midday frame and an overcast one.
    return cv2.normalize(tophat, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def extract_markings(image: np.ndarray) -> MarkingField:
    """Find candidate marking segments across a whole frame."""
    response = paint_response(image)
    lines = cv2.HoughLinesP(
        cv2.Canny(response, 40, 120, apertureSize=3),
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP,
    )
    if lines is None:
        return MarkingField(response=response)
    return MarkingField(segments=lines.reshape(-1, 4).astype(np.float32), response=response)


def _line_through(start: np.ndarray, end: np.ndarray) -> tuple[float, float, float]:
    """Line as ``ax + by = c`` with a unit normal, so ``c`` is a real distance."""
    direction = end - start
    length = float(np.hypot(*direction))
    if length < 1e-6:
        return 0.0, 0.0, 0.0
    normal = np.array([-direction[1], direction[0]]) / length
    return float(normal[0]), float(normal[1]), float(normal @ start)


def _intersect(
    first: tuple[float, float, float], second: tuple[float, float, float]
) -> np.ndarray | None:
    matrix = np.array([[first[0], first[1]], [second[0], second[1]]], dtype=np.float64)
    determinant = float(np.linalg.det(matrix))
    # Near-parallel edges intersect somewhere useless; leave the corner alone.
    if abs(determinant) < 1e-3:
        return None
    return np.linalg.solve(matrix, np.array([first[2], second[2]], dtype=np.float64))


def _support_for_edge(
    start: np.ndarray, end: np.ndarray, field_segments: np.ndarray, scale: float
) -> tuple[tuple[float, float, float] | None, float]:
    """Best supporting line for one bay edge, and how much of it is covered."""
    edge_vector = end - start
    edge_length = float(np.hypot(*edge_vector))
    if edge_length < 1e-6 or field_segments.shape[0] == 0:
        return None, 0.0
    edge_angle = np.degrees(np.arctan2(edge_vector[1], edge_vector[0])) % 180.0
    normal_a, normal_b, offset = _line_through(start, end)

    starts = field_segments[:, :2]
    ends = field_segments[:, 2:]
    vectors = ends - starts
    lengths = np.hypot(vectors[:, 0], vectors[:, 1])
    valid = lengths > 1e-6
    if not np.any(valid):
        return None, 0.0

    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0])) % 180.0
    angle_delta = np.abs(angles - edge_angle)
    angle_delta = np.minimum(angle_delta, 180.0 - angle_delta)

    midpoints = (starts + ends) / 2.0
    distance = np.abs(midpoints @ np.array([normal_a, normal_b]) - offset)

    # Signed offset, so a stripe in front of the edge and one behind it land in
    # different groups instead of averaging to the edge's current position.
    signed = midpoints @ np.array([normal_a, normal_b]) - offset
    accepted = valid & (angle_delta <= MAX_ANGLE_DELTA_DEGREES)
    accepted &= distance <= MAX_OFFSET_RATIO * scale
    if not np.any(accepted):
        return None, 0.0

    unit = edge_vector / edge_length
    indices = np.nonzero(accepted)[0]
    order = indices[np.argsort(signed[indices])]

    # Walk the offsets in order and cut a new group wherever the gap exceeds the
    # clustering tolerance; each group is then one painted stripe.
    groups: list[list[int]] = [[int(order[0])]]
    tolerance = OFFSET_CLUSTER_RATIO * scale
    for index in order[1:]:
        if signed[index] - signed[groups[-1][-1]] > tolerance:
            groups.append([int(index)])
        else:
            groups[-1].append(int(index))

    best_support = 0.0
    best_group: list[int] | None = None
    for group in groups:
        support = _covered_fraction(group, starts, ends, start, unit, edge_length)
        if support > best_support:
            best_support, best_group = support, group

    if best_group is None or best_support < MIN_EDGE_SUPPORT:
        return None, best_support

    # Fit to the winning stripe only, keeping the bay edge's own direction so a
    # stray segment can reposition the boundary but never rotate it.
    weights = lengths[best_group]
    centre = np.average(midpoints[best_group], axis=0, weights=weights)
    normal = np.array([-unit[1], unit[0]])
    shift = abs(float(normal @ centre) - float(normal @ start))
    if shift > MAX_SHIFT_RATIO * scale:
        return None, best_support
    return (float(normal[0]), float(normal[1]), float(normal @ centre)), best_support


def _covered_fraction(
    group: list[int],
    starts: np.ndarray,
    ends: np.ndarray,
    origin: np.ndarray,
    unit: np.ndarray,
    edge_length: float,
) -> float:
    """Share of an edge covered by a group of segments projected onto it.

    Several short collinear fragments of one painted stripe should count the
    same as one long segment, so overlapping projections are merged rather than
    summed.
    """
    spans: list[tuple[float, float]] = []
    for index in group:
        projections = (
            float((starts[index] - origin) @ unit),
            float((ends[index] - origin) @ unit),
        )
        low = max(min(projections), 0.0)
        high = min(max(projections), edge_length)
        if high > low:
            spans.append((low, high))
    if not spans:
        return 0.0
    spans.sort()
    total = spans[0][1] - spans[0][0]
    reach = spans[0][1]
    for low, high in spans[1:]:
        if low > reach:
            total += high - low
        elif high > reach:
            total += high - reach
        reach = max(reach, high)
    return float(total / edge_length)


def refine_quad(
    markings: MarkingField, polygon: list[list[float]], width: int, height: int
) -> RefinedQuad:
    """Pull a proposed bay onto the painted markings that support it.

    ``polygon`` is normalised and clockwise; the result is returned the same way
    so callers can substitute it directly.
    """
    original = np.array([[x * width, y * height] for x, y in polygon], dtype=np.float64)
    if original.shape != (4, 2) or markings.empty:
        return RefinedQuad(polygon, False, 0, 0.0, "no marking evidence available")

    area = abs(float(cv2.contourArea(original.astype(np.float32))))
    if area <= 1.0:
        return RefinedQuad(polygon, False, 0, 0.0, "degenerate proposal")
    scale = float(np.sqrt(area))

    lines: list[tuple[float, float, float] | None] = []
    supports: list[float] = []
    for index in range(4):
        start, end = original[index], original[(index + 1) % 4]
        line, support = _support_for_edge(start, end, markings.segments, scale)
        lines.append(line)
        supports.append(support)

    snapped = sum(1 for line in lines if line is not None)
    mean_support = float(np.mean(supports))
    if snapped < MIN_SNAPPED_EDGES:
        return RefinedQuad(
            polygon, False, snapped, mean_support, "too few edges met the support threshold"
        )

    # Unsupported edges keep their original line, so a bay with two visible
    # stripes is still improved along those two without inventing the others.
    for index in range(4):
        if lines[index] is None:
            lines[index] = _line_through(original[index], original[(index + 1) % 4])

    corners = []
    for index in range(4):
        previous = lines[index - 1]
        current = lines[index]
        assert previous is not None and current is not None
        point = _intersect(previous, current)
        if point is None:
            return RefinedQuad(polygon, False, snapped, mean_support, "edges became parallel")
        corners.append(point)
    refined = np.array(corners, dtype=np.float32)

    guard = _guard(original.astype(np.float32), refined, width, height)
    if guard:
        return RefinedQuad(polygon, False, snapped, mean_support, guard)

    # The decisive test.  A Hough segment only proves something linear is there;
    # tarmac cracks, shadow boundaries and vehicle sills all qualify.  Paint is
    # what is *bright*, so the refined outline is only accepted if it sits on
    # more marking response than the proposal it would replace.
    before = _outline_brightness(markings, original.astype(np.float32))
    after = _outline_brightness(markings, refined)
    if after < before + MIN_BRIGHTNESS_GAIN:
        return RefinedQuad(
            polygon, False, snapped, mean_support, "refined outline sits on no brighter marking"
        )

    return RefinedQuad(
        polygon=[
            [float(np.clip(x / width, 0.0, 1.0)), float(np.clip(y / height, 0.0, 1.0))]
            for x, y in refined
        ],
        refined=True,
        edges_snapped=snapped,
        support=mean_support,
        reason=f"snapped {snapped} edge(s) to painted markings",
    )


def _outline_brightness(markings: MarkingField, quad: np.ndarray) -> float:
    """Mean marking response along a quadrilateral's four edges."""
    return float(
        np.mean(
            [
                markings.brightness_along(quad[index], quad[(index + 1) % 4])
                for index in range(4)
            ]
        )
    )


def _guard(original: np.ndarray, refined: np.ndarray, width: int, height: int) -> str:
    """Reject a refinement that moved the bay rather than corrected it."""
    refined_area = abs(float(cv2.contourArea(refined)))
    original_area = abs(float(cv2.contourArea(original)))
    if refined_area <= 1.0:
        return "refinement collapsed the bay"
    ratio = refined_area / original_area
    if ratio > MAX_AREA_RATIO or ratio < 1.0 / MAX_AREA_RATIO:
        return f"refinement changed area by {ratio:.2f}x"
    if np.any(refined[:, 0] < -width * 0.05) or np.any(refined[:, 0] > width * 1.05):
        return "refinement left the frame"
    if np.any(refined[:, 1] < -height * 0.05) or np.any(refined[:, 1] > height * 1.05):
        return "refinement left the frame"
    intersection, _ = cv2.intersectConvexConvex(original, refined)
    union = original_area + refined_area - float(intersection)
    if union <= 0 or float(intersection) / union < MIN_REFINED_IOU:
        return "refinement moved the bay too far"
    return ""


def scene_response(markings: MarkingField, polygons: list[list[list[float]]],
                   width: int, height: int) -> float:
    """How strongly marked the parking surface is, independent of the proposals.

    Sampled over the ground the proposals cover rather than the whole frame, so
    a car park ringed by bright buildings does not read as well marked, and
    *inside* the regions rather than along their outlines, because the outline
    is exactly what is still wrong at this point -- gating on it would ask the
    proposal to prove itself before being allowed to be corrected.

    The measure is the upper decile of the response inside those regions.
    Painted stripes occupy a small share of a bay's area, so a plain mean is
    dominated by bare tarmac and cannot separate a crisply marked surface from
    an unmarked one.
    """
    if not polygons or markings.response is None:
        return 0.0
    mask = np.zeros(markings.response.shape[:2], dtype=np.uint8)
    for polygon in polygons:
        quad = np.array(
            [[x * width, y * height] for x, y in polygon], dtype=np.float32
        )
        if quad.shape != (4, 2):
            continue
        cv2.fillConvexPoly(mask, quad.astype(np.int32), 1)
    values = markings.response[mask.astype(bool)]
    if values.size < 32:
        return 0.0
    return float(np.mean(values[values >= np.percentile(values, 90)])) / 255.0


def refine_all(
    image: np.ndarray, polygons: list[list[list[float]]]
) -> list[RefinedQuad]:
    """Refine every bay in one frame against one shared marking extraction.

    Returns the proposals untouched when the scene carries too little marking
    response to refine against, which is the common case for older CCTV car
    parks with worn or absent paint.
    """
    if not polygons:
        return []
    height, width = image.shape[:2]
    markings = extract_markings(image)
    response = scene_response(markings, polygons, width, height)
    if response < MIN_SCENE_RESPONSE:
        note = f"scene marking response {response:.3f} below the refinement floor"
        return [RefinedQuad(polygon, False, 0, response, note) for polygon in polygons]
    return [refine_quad(markings, polygon, width, height) for polygon in polygons]
