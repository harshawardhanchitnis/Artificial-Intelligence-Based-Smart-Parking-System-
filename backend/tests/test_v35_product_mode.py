"""Regression tests for the product-grade scene understanding work.

Each test names the behaviour it locks down.  The emphasis is on the cases the
audit measured going wrong -- geometry that is not a rectangle, vehicles that
exist outside any known bay, and evidence that must inform a verdict without
being allowed to dictate it.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.datasets.quad_dataset import (
    POSE_FLIP_INDEX,
    canonical_quad,
    obb_label,
    pose_label,
    seg_label,
)
from app.ml.association import (
    GROUND_BAND,
    OCCUPANCY_OVERLAP,
    associate,
    coverage,
    ground_footprint,
)
from app.ml.line_refinement import MarkingField, refine_all, refine_quad
from app.ml.occupancy_fusion import (
    DEFAULT_COEFFICIENTS,
    FusionPolicy,
    OccupancyEvidence,
    default_policy,
)
from app.ml.parking_area import (
    AREA_UNKNOWN,
    IN_MAPPED_SPACE,
    IN_PARKING_AREA,
    OUTSIDE_PARKING_AREA,
    ParkingArea,
    classify_vehicle_positions,
    infer_parking_area,
    summarise,
)
from app.ml.vehicle_detector import (
    VehicleDetection,
    non_max_suppression,
    unclassified_count,
    vehicle_counts,
)
from app.ml.vehicle_taxonomy import (
    CAR,
    PRODUCT_CLASSES,
    TRUCK,
    TWO_WHEELER,
    display_name,
    product_class,
)
from app.ml.vehicle_tracking import VehicleTracker
from app.services.auto_calibration import (
    ACTIVE,
    MIN_LAYOUT_SLOTS,
    UNRESOLVED,
    BayTrack,
    CalibrationResult,
)
from app.services.localization_service import layout_agreement
from app.services.scene_analysis import BENCHMARK_MODE, PRODUCT_MODE, SceneAnalysis, SceneSlot

# --------------------------------------------------------------------------
# Four-corner geometry -- a bay is a quadrilateral, not a rectangle
# --------------------------------------------------------------------------

# A bay seen down a row: near edge long, far edge short.  A rotated rectangle
# cannot represent this, which is the whole reason for the pose representation.
_PERSPECTIVE_BAY = [[0.10, 0.60], [0.40, 0.55], [0.46, 0.90], [0.06, 0.95]]

# A bay in an angled parking row.
_DIAGONAL_BAY = [[0.30, 0.20], [0.52, 0.28], [0.44, 0.52], [0.22, 0.44]]


def test_arbitrary_quadrilateral_survives_the_pose_label_round_trip() -> None:
    quad = canonical_quad(_PERSPECTIVE_BAY)
    assert quad is not None
    values = [float(value) for value in pose_label(quad).split()[1:]]
    # Class, four box values, then four (x, y, visibility) triples.
    assert len(values) == 4 + 12
    corners = [values[4 + index * 3] for index in range(4)] + [
        values[5 + index * 3] for index in range(4)
    ]
    expected = [point[0] for point in quad] + [point[1] for point in quad]
    assert corners == pytest.approx(expected, abs=1e-6)


def test_pose_label_box_encloses_the_quadrilateral() -> None:
    quad = canonical_quad(_DIAGONAL_BAY)
    assert quad is not None
    centre_x, centre_y, width, height = (float(v) for v in pose_label(quad).split()[1:5])
    xs = [point[0] for point in quad]
    ys = [point[1] for point in quad]
    assert centre_x - width / 2 == pytest.approx(min(xs), abs=1e-6)
    assert centre_y + height / 2 == pytest.approx(max(ys), abs=1e-6)


def test_strong_perspective_bay_is_not_a_rectangle_after_canonicalisation() -> None:
    """The representation must preserve the trapezoid, not square it up."""
    quad = canonical_quad(_PERSPECTIVE_BAY)
    assert quad is not None
    near = np.linalg.norm(np.array(quad[2]) - np.array(quad[3]))
    far = np.linalg.norm(np.array(quad[0]) - np.array(quad[1]))
    # The two parallel-ish edges differ in length; a rectangle's would not.
    assert abs(near - far) > 0.05


def test_canonical_quad_orders_corners_the_same_way_regardless_of_input_order() -> None:
    """Keypoint one must mean the same corner in every image, or nothing trains."""
    first = canonical_quad(_DIAGONAL_BAY)
    rotated = canonical_quad(_DIAGONAL_BAY[2:] + _DIAGONAL_BAY[:2])
    assert first == rotated


def test_canonical_quad_rejects_malformed_annotations() -> None:
    assert canonical_quad([[0.1, 0.1], [0.2, 0.2], [0.3, 0.3]]) is None
    assert canonical_quad([[0.1, 0.1], [1.4, 0.2], [0.3, 0.3], [0.2, 0.5]]) is None
    # Degenerate: zero height.
    assert canonical_quad([[0.1, 0.5], [0.4, 0.5], [0.4, 0.5], [0.1, 0.5]]) is None


def test_pose_flip_index_maps_corners_across_a_horizontal_flip() -> None:
    """Clockwise from top-left means a flip swaps left for right on each edge."""
    assert POSE_FLIP_INDEX == (1, 0, 3, 2)


def test_seg_and_obb_labels_carry_the_same_four_corners() -> None:
    quad = canonical_quad(_PERSPECTIVE_BAY)
    assert quad is not None
    assert seg_label(quad).split()[1:] == obb_label(quad).split()[1:]


# --------------------------------------------------------------------------
# Vehicle taxonomy -- exactly three user-facing classes, and no more
# --------------------------------------------------------------------------


def test_only_three_product_vehicle_classes_exist() -> None:
    assert PRODUCT_CLASSES == (CAR, TWO_WHEELER, TRUCK)
    assert len(PRODUCT_CLASSES) == 3


@pytest.mark.parametrize(
    ("source", "expected"),
    [
        ("car", CAR),
        ("suv", CAR),
        ("van", CAR),
        ("minivan", CAR),
        ("sedan", CAR),
        ("hatchback", CAR),
        ("motorcycle", TWO_WHEELER),
        ("motorbike", TWO_WHEELER),
        ("scooter", TWO_WHEELER),
        ("moped", TWO_WHEELER),
        ("truck", TRUCK),
        ("lorry", TRUCK),
        ("pickup", TRUCK),
    ],
)
def test_source_labels_fold_into_the_three_product_classes(source: str, expected: str) -> None:
    assert product_class(source) == expected


@pytest.mark.parametrize(
    "source", ["person", "traffic light", "bench", "dog", "boat", "train", "aeroplane"]
)
def test_non_vehicles_never_become_a_fourth_class(source: str) -> None:
    assert product_class(source) is None


@pytest.mark.parametrize("source", ["bus", "bicycle"])
def test_excluded_vehicles_are_unsupported_not_reclassified(source: str) -> None:
    """A coach is not a lorry and a pedal cycle is not a motorcycle.

    Folding ``bus`` into ``TRUCK`` put an unsupported vehicle in front of the
    user under a supported name, and on one steep top-down view it surfaced
    31 ordinary cars as trucks.
    """
    assert product_class(source) is None


def test_bus_is_never_mapped_to_truck_by_any_route() -> None:
    from app.ml.vehicle_taxonomy import EXTENDED_CLASS_MAP, UNSUPPORTED_VEHICLE_LABELS

    assert "bus" not in EXTENDED_CLASS_MAP
    assert "bus" in UNSUPPORTED_VEHICLE_LABELS
    assert TRUCK not in {product_class("bus"), product_class("coach")}


def test_every_mapped_label_lands_in_the_product_taxonomy() -> None:
    """No mapping may introduce a class the interface does not know about."""
    from app.ml.vehicle_taxonomy import EXTENDED_CLASS_MAP

    assert set(EXTENDED_CLASS_MAP.values()) <= set(PRODUCT_CLASSES)


def test_display_names_exist_for_every_product_class() -> None:
    assert [display_name(name) for name in PRODUCT_CLASSES] == ["Car", "Two-wheeler", "Truck"]


def test_vehicle_counts_always_reports_all_three_classes() -> None:
    assert vehicle_counts([]) == {CAR: 0, TWO_WHEELER: 0, TRUCK: 0}


def test_source_label_casing_does_not_leak_a_new_class() -> None:
    assert product_class("Motorcycle") == TWO_WHEELER
    assert product_class(" TRUCK ") == TRUCK


# --------------------------------------------------------------------------
# Slot / vehicle association -- IoU alone gets the common cases backwards
# --------------------------------------------------------------------------

# A 100x200 bay at the origin, in pixels.
_BAY = [[0.0, 0.0], [100.0, 0.0], [100.0, 200.0], [0.0, 200.0]]
_NEXT_BAY = [[100.0, 0.0], [200.0, 0.0], [200.0, 200.0], [100.0, 200.0]]


def test_ground_footprint_uses_only_the_lower_band_of_a_vehicle_box() -> None:
    """A box encloses the vehicle's height; only its base meets the ground."""
    footprint = ground_footprint((0.0, 0.0, 100.0, 200.0))
    assert footprint[:, 1].min() == pytest.approx(200.0 - 200.0 * GROUND_BAND)
    assert footprint[:, 1].max() == pytest.approx(200.0)


def test_a_car_filling_one_bay_occupies_exactly_that_bay() -> None:
    result = associate([_BAY, _NEXT_BAY], [(10.0, 20.0, 90.0, 190.0)])
    assert [link.slot_index for link in result.links if link.occupies] == [0]
    assert result.unmapped_vehicles == []


def test_a_lorry_straddling_two_bays_occupies_both() -> None:
    """Neither bay reaches a high IoU with the vehicle, yet both are filled."""
    result = associate([_BAY, _NEXT_BAY], [(20.0, 10.0, 180.0, 195.0)])
    occupied = sorted(link.slot_index for link in result.links if link.occupies)
    assert occupied == [0, 1]


def test_a_two_wheeler_occupies_a_full_size_bay_it_barely_covers() -> None:
    """Covers ~7% of the bay: IoU would dismiss it, the product must not."""
    result = associate([_BAY], [(40.0, 140.0, 65.0, 190.0)])
    links = [link for link in result.links if link.occupies]
    assert len(links) == 1
    # The bay-side ratio is far below the threshold; only the vehicle-side
    # ratio makes this bay occupied.
    assert links[0].slot_coverage < OCCUPANCY_OVERLAP
    assert links[0].vehicle_coverage > 0.9


def test_two_two_wheelers_can_share_one_bay() -> None:
    result = associate([_BAY], [(20.0, 140.0, 45.0, 190.0), (55.0, 140.0, 80.0, 190.0)])
    assert len({link.vehicle_index for link in result.links if link.occupies}) == 2


def test_a_vehicle_in_the_aisle_is_unmapped_rather_than_occupying_a_bay() -> None:
    result = associate([_BAY], [(400.0, 400.0, 480.0, 560.0)])
    assert result.unmapped_vehicles == [0]
    assert [link for link in result.links if link.occupies] == []


def test_a_vehicle_clipping_a_bay_corner_does_not_occupy_it() -> None:
    """Overlaps the bay slightly but sits mostly outside it."""
    result = associate([_BAY], [(80.0, 180.0, 300.0, 400.0)])
    assert [link for link in result.links if link.occupies] == []
    assert result.unmapped_vehicles == [0]


def test_coverage_reports_both_directions_independently() -> None:
    bay = np.array(_BAY, dtype=np.float32)
    small = np.array([[40.0, 40.0], [60.0, 40.0], [60.0, 80.0], [40.0, 80.0]], dtype=np.float32)
    _, bay_share, vehicle_share = coverage(bay, small)
    assert bay_share < 0.10
    assert vehicle_share == pytest.approx(1.0, abs=1e-3)


def test_unmapped_vehicles_are_reported_when_no_bay_was_detected_at_all() -> None:
    """Vehicle understanding must not depend on bay detection succeeding."""
    result = associate([], [(10.0, 10.0, 90.0, 190.0), (200.0, 10.0, 280.0, 190.0)])
    assert result.unmapped_vehicles == [0, 1]


# --------------------------------------------------------------------------
# Evidence fusion -- signals inform the verdict, none of them dictates it
# --------------------------------------------------------------------------


def _policy(**overrides: float) -> FusionPolicy:
    coefficients = dict(DEFAULT_COEFFICIENTS)
    coefficients.update(overrides)
    return FusionPolicy(
        coefficients=coefficients,
        lower_threshold=0.35,
        upper_threshold=0.65,
        metadata={"fitted": True},
    )


def test_without_a_fitted_artifact_fusion_reproduces_the_classifier() -> None:
    """A machine with no fusion artifact must behave exactly as it did before."""
    policy = default_policy()
    for probability in (0.02, 0.3, 0.5, 0.8, 0.99):
        fused = policy.fuse(OccupancyEvidence(occupancy_probability=probability))
        assert fused.probability == pytest.approx(probability, abs=1e-6)


def test_a_missing_vehicle_detection_cannot_veto_a_confident_classifier() -> None:
    """A motorcycle behind a van is invisible to the detector and real to the crop."""
    policy = _policy(vehicle_present=2.0, vehicle_absent=-0.8)
    fused = policy.fuse(
        OccupancyEvidence(
            occupancy_probability=0.985,
            vehicle_evidence_available=True,
            vehicle_detected=False,
        )
    )
    assert fused.state == "occupied"


def test_an_unrecognised_object_still_leaves_a_bay_occupied_or_uncertain() -> None:
    """Occupancy must not require the object to be one of the three classes."""
    policy = _policy(vehicle_present=2.0, vehicle_absent=-0.8)
    fused = policy.fuse(
        OccupancyEvidence(
            occupancy_probability=0.80,
            vehicle_evidence_available=True,
            vehicle_detected=False,
        )
    )
    assert fused.state in {"occupied", "uncertain"}
    assert fused.state != "vacant"


def test_a_vehicle_covering_a_bay_prevents_a_confident_vacant_verdict() -> None:
    policy = _policy(vehicle_present=2.5, slot_coverage=2.0)
    fused = policy.fuse(
        OccupancyEvidence(
            occupancy_probability=0.30,
            vehicle_evidence_available=True,
            vehicle_detected=True,
            vehicle_confidence=0.9,
            slot_coverage=0.85,
            vehicle_coverage=0.9,
        )
    )
    assert fused.state != "vacant"


def test_absent_vehicle_evidence_differs_from_an_absent_detector() -> None:
    """"No detector ran" must not be read as "no vehicle is there"."""
    policy = _policy(vehicle_absent=-1.5)
    no_detector = policy.fuse(OccupancyEvidence(occupancy_probability=0.7))
    detector_saw_nothing = policy.fuse(
        OccupancyEvidence(
            occupancy_probability=0.7, vehicle_evidence_available=True, vehicle_detected=False
        )
    )
    assert no_detector.probability > detector_saw_nothing.probability


def test_weak_evidence_lands_in_the_uncertain_band() -> None:
    policy = _policy()
    assert policy.fuse(OccupancyEvidence(occupancy_probability=0.5)).state == "uncertain"


def test_an_uncertain_bay_is_never_counted_as_occupied() -> None:
    policy = _policy()
    fused = policy.fuse(OccupancyEvidence(occupancy_probability=0.5))
    assert fused.state == "uncertain"
    assert fused.occupied is False
    assert fused.as_dict()["predicted_occupied"] is False


def test_doubtful_geometry_pulls_a_verdict_toward_uncertain() -> None:
    policy = _policy(geometry_confidence=1.5)
    confident = policy.fuse(
        OccupancyEvidence(occupancy_probability=0.72, geometry_confidence=1.0)
    )
    doubtful = policy.fuse(
        OccupancyEvidence(occupancy_probability=0.72, geometry_confidence=0.4)
    )
    assert doubtful.probability < confident.probability


# --------------------------------------------------------------------------
# Automatic calibration -- consensus across frames, and honest abstention
# --------------------------------------------------------------------------


def test_a_bay_seen_in_every_frame_averages_toward_its_true_position() -> None:
    track = BayTrack()
    truth = np.array(_DIAGONAL_BAY)
    rng = np.random.default_rng(11)
    for _ in range(9):
        track.corners.append(truth + rng.normal(0, 0.004, truth.shape))
        track.confidences.append(0.7)
    averaged = np.array(track.mean_polygon())
    single = track.corners[0]
    assert np.abs(averaged - truth).mean() < np.abs(single - truth).mean()


def test_calibration_reports_unresolved_rather_than_inventing_a_layout() -> None:
    result = CalibrationResult(UNRESOLVED, frames_used=7, message="nothing recurred")
    assert result.established is False
    assert result.as_dict()["calibration_state"] == UNRESOLVED


def test_a_layout_below_the_plausible_minimum_is_not_established() -> None:
    slots = [{"polygon": _BAY} for _ in range(MIN_LAYOUT_SLOTS - 1)]
    result = CalibrationResult(ACTIVE, slots=slots, frames_used=7, consensus=0.9)
    assert result.established is False


def test_an_established_layout_needs_no_human_confirmation() -> None:
    slots = [{"polygon": _BAY} for _ in range(MIN_LAYOUT_SLOTS + 2)]
    result = CalibrationResult(ACTIVE, slots=slots, frames_used=7, consensus=0.85)
    assert result.established is True
    assert result.state == ACTIVE


# --------------------------------------------------------------------------
# Benchmark mode against product mode -- kept separate on purpose
# --------------------------------------------------------------------------


def _slot(state: str) -> SceneSlot:
    return SceneSlot(
        index=1,
        polygon=_DIAGONAL_BAY,
        state=state,
        occupied_probability=0.9 if state == "occupied" else 0.1,
        confidence=0.9,
        classifier_probability=0.9 if state == "occupied" else 0.1,
    )


def test_benchmark_mode_reports_no_vehicles_and_no_fusion() -> None:
    """Historical figures were produced without either; comparability requires it."""
    analysis = SceneAnalysis(mode=BENCHMARK_MODE, slots=[_slot("occupied")])
    payload = analysis.as_dict()
    assert payload["analysis_mode"] == BENCHMARK_MODE
    assert "vehicles" not in payload
    assert "unmapped_vehicles" not in payload
    assert payload["fusion_applied"] is False


def test_product_mode_separates_site_vehicles_from_passing_traffic() -> None:
    """"Unmapped" means inside the facility, not merely outside a bay.

    Counting every unassociated detection as unmapped let a road behind the
    site inflate its figures: one CNRPark view reported 40 unmapped vehicles,
    most of them traffic.
    """
    analysis = SceneAnalysis(
        mode=PRODUCT_MODE,
        slots=[_slot("occupied")],
        vehicles=[
            _object(CAR, "car", 0.0),
            _object(CAR, "car", 0.1),
            _object(CAR, "car", 0.2),
        ],
        placements=[IN_MAPPED_SPACE, IN_PARKING_AREA, OUTSIDE_PARKING_AREA],
        vehicle_detection_available=True,
    )
    payload = analysis.as_dict()
    assert payload["unmapped_vehicles"] == 1
    assert payload["vehicles_outside_parking_area"] == 1
    # Class counts describe the facility: the vehicle on the road is excluded.
    assert payload["vehicle_counts"] == {CAR: 2, TWO_WHEELER: 0, TRUCK: 0}
    assert payload["scene_vehicle_counts"] == {CAR: 3, TWO_WHEELER: 0, TRUCK: 0}


def test_space_counts_always_reconcile_across_the_three_states() -> None:
    analysis = SceneAnalysis(
        mode=PRODUCT_MODE,
        slots=[_slot("occupied"), _slot("vacant"), _slot("uncertain"), _slot("vacant")],
    )
    payload = analysis.as_dict()
    assert payload["occupied_spaces"] == 1
    assert payload["uncertain_spaces"] == 1
    assert payload["vacant_spaces"] == 2
    total = (
        payload["occupied_spaces"] + payload["vacant_spaces"] + payload["uncertain_spaces"]
    )
    assert total == payload["total_spaces"]


# --------------------------------------------------------------------------
# Marking refinement -- guarded, and silent when there is nothing to use
# --------------------------------------------------------------------------


def test_refinement_returns_the_proposal_untouched_without_marking_evidence() -> None:
    result = refine_quad(MarkingField(), _DIAGONAL_BAY, 640, 480)
    assert result.refined is False
    assert result.polygon == _DIAGONAL_BAY


def test_refinement_declines_on_a_blank_surface_rather_than_inventing_edges() -> None:
    blank = np.full((480, 640, 3), 90, dtype=np.uint8)
    results = refine_all(blank, [_DIAGONAL_BAY, _PERSPECTIVE_BAY])
    assert [item.refined for item in results] == [False, False]
    assert [item.polygon for item in results] == [_DIAGONAL_BAY, _PERSPECTIVE_BAY]


def test_refinement_is_a_no_op_when_no_polygons_are_supplied() -> None:
    assert refine_all(np.zeros((32, 32, 3), dtype=np.uint8), []) == []


# --------------------------------------------------------------------------
# Vehicle detector decoding
# --------------------------------------------------------------------------


def test_non_max_suppression_keeps_the_strongest_of_overlapping_boxes() -> None:
    boxes = np.array(
        [[0, 0, 100, 100], [5, 5, 105, 105], [400, 400, 500, 500]], dtype=np.float32
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    assert sorted(non_max_suppression(boxes, scores, 0.5)) == [0, 2]


def test_non_max_suppression_keeps_neighbouring_vehicles_apart() -> None:
    """Cars parked side by side must not suppress one another."""
    boxes = np.array([[0, 0, 100, 200], [110, 0, 210, 200]], dtype=np.float32)
    scores = np.array([0.9, 0.85], dtype=np.float32)
    assert sorted(non_max_suppression(boxes, scores, 0.5)) == [0, 1]


# --------------------------------------------------------------------------
# Video accounting -- the three states must partition the layout exactly
# --------------------------------------------------------------------------


def _frame_states(smoothed: list[float], hysteresis: list[bool], band: tuple[float, float]):
    """The rule the video pipeline uses to turn two signals into one verdict."""
    lower, upper = band
    return [
        "uncertain"
        if lower <= probability <= upper
        else ("occupied" if state else "vacant")
        for probability, state in zip(smoothed, hysteresis, strict=True)
    ]


def test_video_frame_states_partition_the_layout() -> None:
    """Occupied + vacant + uncertain must equal the slot count on every frame.

    The counts were previously derived from two independent rules -- a
    hysteresis flag for occupied and a band test for uncertain -- so a bay above
    the threshold but inside the band was counted twice and the vacant total was
    computed by subtracting both.
    """
    smoothed = [0.02, 0.45, 0.50, 0.55, 0.97, 0.30, 0.80]
    hysteresis = [False, False, True, True, True, False, True]
    states = _frame_states(smoothed, hysteresis, (0.35, 0.56))
    occupied = states.count("occupied")
    vacant = states.count("vacant")
    uncertain = states.count("uncertain")
    assert occupied + vacant + uncertain == len(smoothed)
    # 0.45, 0.50 and 0.55 all sit inside the band regardless of hysteresis.
    assert uncertain == 3


def test_a_bay_inside_the_band_is_uncertain_even_when_hysteresis_says_occupied() -> None:
    states = _frame_states([0.55], [True], (0.35, 0.56))
    assert states == ["uncertain"]


def test_a_bay_outside_the_band_follows_hysteresis_not_the_raw_threshold() -> None:
    """Hysteresis is what stops a bay flickering around the decision point."""
    assert _frame_states([0.60], [False], (0.35, 0.56)) == ["vacant"]
    assert _frame_states([0.60], [True], (0.35, 0.56)) == ["occupied"]



# --------------------------------------------------------------------------
# Unclassified objects -- evidence without a class
# --------------------------------------------------------------------------


def _object(product: str | None, source: str, x: float = 0.0) -> VehicleDetection:
    return VehicleDetection(
        product_class=product, confidence=0.8, box=(x, 0.0, x + 0.05, 0.05), source_label=source
    )


def test_unclassified_objects_are_kept_but_never_counted_as_a_class() -> None:
    detections = [
        _object(CAR, "car", 0.0),
        _object(None, "bus", 0.1),
        _object(None, "bus", 0.2),
        _object(TRUCK, "truck", 0.3),
    ]
    assert vehicle_counts(detections) == {CAR: 1, TWO_WHEELER: 0, TRUCK: 1}
    assert unclassified_count(detections) == 2
    # They still exist, so they can still carry occupancy evidence.
    assert len(detections) == 4


def test_class_counts_never_sum_to_more_than_the_classified_detections() -> None:
    detections = [_object(None, "bus", index * 0.06) for index in range(5)]
    assert sum(vehicle_counts(detections).values()) == 0
    assert unclassified_count(detections) == 5


def test_an_unclassified_object_can_still_occupy_a_bay() -> None:
    """A coach fills a bay whether or not the taxonomy has a name for it."""
    result = associate([_BAY], [(10.0, 20.0, 90.0, 190.0)])
    assert [link.slot_index for link in result.links if link.occupies] == [0]


# --------------------------------------------------------------------------
# Parking-area understanding -- road traffic must not inflate site figures
# --------------------------------------------------------------------------


def _grid_bays(rows: int, columns: int, size: float = 0.05, gap: float = 0.06):
    """A small regular car park in normalised coordinates."""
    return [
        [
            [0.1 + column * gap, 0.4 + row * gap],
            [0.1 + column * gap + size, 0.4 + row * gap],
            [0.1 + column * gap + size, 0.4 + row * gap + size],
            [0.1 + column * gap, 0.4 + row * gap + size],
        ]
        for row in range(rows)
        for column in range(columns)
    ]


def test_no_bays_means_no_parking_area_rather_than_a_guess() -> None:
    assert infer_parking_area([]).known is False


def test_the_area_covers_the_bays_that_define_it() -> None:
    bays = _grid_bays(2, 4)
    area = infer_parking_area(bays)
    assert area.known
    for bay in bays:
        centre_x = sum(point[0] for point in bay) / 4
        centre_y = sum(point[1] for point in bay) / 4
        assert area.contains(centre_x, centre_y)


def test_the_area_bridges_the_aisle_between_two_rows() -> None:
    """An aisle has bays on both sides; a road does not."""
    near = _grid_bays(1, 4)
    far = [
        [[x, y + 0.16] for x, y in bay]  # a row one aisle away
        for bay in _grid_bays(1, 4)
    ]
    area = infer_parking_area(near + far)
    # A point in the aisle, between the two rows.
    assert area.contains(0.20, 0.50)


def test_a_vehicle_far_from_the_site_is_outside_the_parking_area() -> None:
    area = infer_parking_area(_grid_bays(2, 4))
    placements = classify_vehicle_positions(area, [(0.80, 0.05, 0.88, 0.11)], set())
    assert placements == [OUTSIDE_PARKING_AREA]


def test_a_vehicle_in_an_aisle_is_inside_the_area_but_unmapped() -> None:
    area = infer_parking_area(_grid_bays(1, 4) + [
        [[x, y + 0.16] for x, y in bay] for bay in _grid_bays(1, 4)
    ])
    placements = classify_vehicle_positions(area, [(0.19, 0.47, 0.23, 0.51)], set())
    assert placements == [IN_PARKING_AREA]


def test_a_mapped_vehicle_is_reported_as_mapped_whatever_the_area_says() -> None:
    area = infer_parking_area(_grid_bays(2, 4))
    placements = classify_vehicle_positions(area, [(0.11, 0.41, 0.14, 0.44)], {0})
    assert placements == [IN_MAPPED_SPACE]


def test_placement_summary_accounts_for_every_vehicle() -> None:
    placements = [IN_MAPPED_SPACE, IN_PARKING_AREA, OUTSIDE_PARKING_AREA, IN_MAPPED_SPACE]
    assert sum(summarise(placements).values()) == len(placements)


def test_without_a_layout_position_is_unknown_rather_than_outside() -> None:
    """Absence of a bay map is not evidence that a vehicle is off-site."""
    placements = classify_vehicle_positions(ParkingArea(), [(0.5, 0.5, 0.6, 0.6)], set())
    assert placements == [AREA_UNKNOWN]


# --------------------------------------------------------------------------
# The four fusion cases that decide whether a bay can be called vacant
# --------------------------------------------------------------------------


def _fitted() -> FusionPolicy:
    """Coefficients of the shape the fitted artifact carries."""
    return _policy(
        occupancy_logit=1.04,
        vehicle_present=-1.16,
        vehicle_absent=-0.34,
        slot_coverage=0.17,
        vehicle_coverage=1.46,
        intercept=0.39,
    )


def test_case_a_classifier_occupied_and_vehicle_seen_is_occupied() -> None:
    fused = _fitted().fuse(
        OccupancyEvidence(
            occupancy_probability=0.95,
            vehicle_evidence_available=True,
            vehicle_detected=True,
            vehicle_confidence=0.85,
            slot_coverage=0.7,
            vehicle_coverage=0.9,
        )
    )
    assert fused.state == "occupied"


def test_case_b_classifier_vacant_but_a_vehicle_covers_the_bay_is_not_vacant() -> None:
    """The classifier can be fooled by shadow or glare; the frame is not."""
    fused = _fitted().fuse(
        OccupancyEvidence(
            occupancy_probability=0.30,
            vehicle_evidence_available=True,
            vehicle_detected=True,
            vehicle_confidence=0.9,
            slot_coverage=0.9,
            vehicle_coverage=0.95,
        )
    )
    assert fused.state != "vacant"


def test_case_c_classifier_occupied_with_no_supported_vehicle_is_not_vacant() -> None:
    """A motorcycle behind a van is invisible to the detector and real to the crop."""
    fused = _fitted().fuse(
        OccupancyEvidence(
            occupancy_probability=0.92,
            vehicle_evidence_available=True,
            vehicle_detected=False,
        )
    )
    assert fused.state != "vacant"
    assert fused.state == "occupied"


def test_case_d_an_unsupported_object_filling_a_bay_is_not_vacant() -> None:
    """A coach carries no product class but still fills the bay it stands in.

    The detection reaches the fusion as vehicle evidence; only the class counts
    exclude it.
    """
    fused = _fitted().fuse(
        OccupancyEvidence(
            occupancy_probability=0.88,
            vehicle_evidence_available=True,
            vehicle_detected=True,
            vehicle_confidence=0.8,
            slot_coverage=0.85,
            vehicle_coverage=0.6,
        )
    )
    assert fused.state != "vacant"


def test_no_amount_of_missing_vehicle_evidence_alone_makes_a_bay_vacant() -> None:
    """Absence of a detection is weak evidence and must stay weak."""
    policy = _fitted()
    for probability in (0.75, 0.85, 0.95, 0.99):
        fused = policy.fuse(
            OccupancyEvidence(
                occupancy_probability=probability,
                vehicle_evidence_available=True,
                vehicle_detected=False,
            )
        )
        assert fused.state != "vacant", probability


# --------------------------------------------------------------------------
# Temporal confirmation -- the only evidence a still image does not have
# --------------------------------------------------------------------------


def test_a_vehicle_seen_in_consecutive_frames_is_confirmed() -> None:
    tracker = VehicleTracker()
    box = (0.10, 0.10, 0.20, 0.20)
    tracker.confirm([box])
    assert tracker.confirm([box]) == [True]
    assert tracker.confirm([box]) == [True]


def test_a_one_frame_artefact_is_rejected() -> None:
    """A painted numeral detected once must not become a vehicle."""
    tracker = VehicleTracker()
    real = (0.10, 0.10, 0.20, 0.20)
    tracker.confirm([real])
    tracker.confirm([real])
    artefact = (0.70, 0.70, 0.76, 0.76)
    assert tracker.confirm([real, artefact]) == [True, False]


def test_a_vehicle_survives_a_single_missed_frame() -> None:
    """An occlusion should not un-detect a parked car."""
    tracker = VehicleTracker()
    box = (0.10, 0.10, 0.20, 0.20)
    tracker.confirm([box])
    tracker.confirm([box])
    tracker.confirm([])          # occluded
    assert tracker.confirm([box]) == [True]


def test_early_frames_are_accepted_while_history_fills() -> None:
    """Reporting no vehicles at all for the opening frames is the worse failure."""
    tracker = VehicleTracker()
    assert tracker.confirm([(0.1, 0.1, 0.2, 0.2)]) == [True]


def test_a_vehicle_that_moves_far_between_frames_is_not_the_same_track() -> None:
    tracker = VehicleTracker()
    tracker.confirm([(0.10, 0.10, 0.20, 0.20)])
    tracker.confirm([(0.10, 0.10, 0.20, 0.20)])
    assert tracker.confirm([(0.60, 0.60, 0.70, 0.70)]) == [False]


# --------------------------------------------------------------------------
# Layout corroboration -- the V3 safety fix must survive a detector swap
# --------------------------------------------------------------------------


def test_corroboration_still_separates_correct_from_wrong_layouts() -> None:
    """A recalled layout is trusted only if detections back it up.

    Measured on nine unseen-camera frames, correct layouts score 0.40 to 0.64
    and layouts belonging to other car parks score at most 0.21, so the
    threshold sits between two populations rather than inside one.
    """
    bays = _grid_bays(2, 4)
    recalled = [{"polygon": bay} for bay in bays]
    detected = [{"polygon": bay} for bay in bays]
    assert layout_agreement(recalled, detected) == pytest.approx(1.0)

    elsewhere = [{"polygon": [[x + 0.45, y] for x, y in bay]} for bay in bays]
    assert layout_agreement(elsewhere, detected) == 0.0


def test_corroboration_is_unavailable_rather_than_permissive_on_few_detections() -> None:
    """Two or three boxes cannot corroborate anything, so the ratio is refused."""
    bays = _grid_bays(2, 4)
    recalled = [{"polygon": bay} for bay in bays]
    detected = [{"polygon": bay} for bay in bays[:3]]
    assert layout_agreement(recalled, detected) == 0.0


def test_corroboration_measures_detections_explained_not_spaces_found() -> None:
    """Detector recall varies by camera; explaining what it sees does not.

    A layout of eight bays where the detector finds only five of them still
    scores 1.0, because every detection is accounted for.  Measuring the other
    direction rejected a valid layout at 0.225 during the V3 work.
    """
    bays = _grid_bays(2, 4)
    recalled = [{"polygon": bay} for bay in bays]
    detected = [{"polygon": bay} for bay in bays[:5]]
    assert layout_agreement(recalled, detected) == pytest.approx(1.0)
