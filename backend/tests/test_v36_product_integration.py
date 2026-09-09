"""Regressions for defects that only manual product testing exposed.

Every test here corresponds to something the automated suite passed while the
running product was wrong, which is the failure mode this file exists to close:

* the installed model card still declared ``bus -> TRUCK`` months after the code
  removed it, so the taxonomy was enforced in one place and contradicted in
  another;
* the JSON body of a POST carried no content type, so FastAPI refused every
  layout correction with a bare "Request validation failed";
* an unresolved layout produced no rendered scene at all, hiding the vehicle
  detection that had in fact succeeded;
* a zero vehicle count could not be told apart from a car park that was empty.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from app.core.config import get_settings
from app.ml.vehicle_detector import SceneDiagnostics, VehicleDetection
from app.ml.vehicle_taxonomy import (
    EXTENDED_CLASS_MAP,
    PRODUCT_CLASSES,
    UNSUPPORTED_VEHICLE_LABELS,
    product_class,
)
from app.services.automatic_analysis import _overlay


def model_card() -> dict[str, object]:
    path = get_settings().model_root / "vehicle-detector-v3.json"
    if not path.is_file():
        pytest.skip("vehicle detector is not installed in this environment")
    return json.loads(path.read_text(encoding="utf-8"))


def test_installed_model_card_never_maps_bus_or_bicycle() -> None:
    """The card must agree with the code, not merely sit next to it."""
    mapping = model_card().get("class_mapping", {})
    assert isinstance(mapping, dict)
    for forbidden in ("bus", "bicycle"):
        assert forbidden not in {key.lower() for key in mapping}, (
            f"the installed model card maps {forbidden!r} into the product taxonomy; "
            "the runtime does not, so the card is lying about the model"
        )
    assert set(mapping.values()) <= set(PRODUCT_CLASSES)


def test_model_card_declares_the_same_classes_the_code_does() -> None:
    card = model_card()
    assert list(card["product_classes"]) == list(PRODUCT_CLASSES)
    assert card["class_mapping"] == {
        name: EXTENDED_CLASS_MAP[name.lower()]
        for name in card["class_names"]
        if name.lower() in EXTENDED_CLASS_MAP
    }
    assert set(card["unsupported_vehicle_labels"]) == set(UNSUPPORTED_VEHICLE_LABELS)


def test_no_source_label_can_reach_a_fourth_public_class() -> None:
    for label in [*EXTENDED_CLASS_MAP, *UNSUPPORTED_VEHICLE_LABELS, "sink", "cell phone"]:
        mapped = product_class(label)
        assert mapped is None or mapped in PRODUCT_CLASSES


def test_scene_diagnostics_separate_an_empty_lot_from_a_blind_detector() -> None:
    """Zero vehicles has two very different meanings; the payload must say which."""
    empty_lot = SceneDiagnostics(confident_objects=0, vehicle_shaped_objects=0,
                                 top_non_vehicle_labels=())
    unreadable = SceneDiagnostics(confident_objects=18, vehicle_shaped_objects=0,
                                  top_non_vehicle_labels=("cell phone", "sink"))
    assert empty_lot.as_dict() != unreadable.as_dict()
    assert unreadable.as_dict()["confident_objects"] == 18
    assert unreadable.as_dict()["top_non_vehicle_labels"] == ["cell phone", "sink"]


def test_overlay_labels_supported_vehicles_and_leaves_unsupported_unnamed(tmp_path) -> None:
    """A reader must be able to see the detection, not just a total.

    The unclassified object is drawn too -- it contributed to occupancy, so
    hiding it would make the picture disagree with the decision -- but it must
    never be given one of the three public class names.
    """
    image = Image.new("RGB", (640, 480), (30, 30, 30))
    destination = tmp_path / "overlay.jpg"
    _overlay(
        image,
        [{"polygon": [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]],
          "occupancy_state": "occupied"}],
        destination,
        vehicles=[
            VehicleDetection("CAR", 0.87, (0.12, 0.12, 0.28, 0.28), "car").as_dict(),
            VehicleDetection(None, 0.44, (0.6, 0.6, 0.8, 0.8), "bus").as_dict(),
        ],
        unmapped=set(),
    )
    assert destination.is_file()
    rendered = np.asarray(Image.open(destination).convert("RGB"), dtype=np.int16)
    plain = np.asarray(image, dtype=np.int16)
    # Both objects leave marks, in different parts of the frame.
    assert np.abs(rendered[48:144, 76:180] - plain[48:144, 76:180]).max() > 30
    assert np.abs(rendered[288:384, 384:512] - plain[288:384, 384:512]).max() > 30


def test_overlay_renders_candidates_without_asserting_a_verdict(tmp_path) -> None:
    """The unresolved path still draws a scene, in a neutral colour."""
    image = Image.new("RGB", (640, 480), (30, 30, 30))
    destination = tmp_path / "candidates.jpg"
    _overlay(
        image,
        [],
        destination,
        vehicles=[],
        candidates=[{"polygon": [[0.2, 0.2], [0.5, 0.2], [0.5, 0.5], [0.2, 0.5]]}],
    )
    rendered = np.asarray(Image.open(destination).convert("RGB"), dtype=np.int16)
    patch = rendered[96:240, 128:320]
    # Neutral: no green or red verdict colour dominates the candidate outline.
    assert abs(int(patch[..., 0].mean()) - int(patch[..., 1].mean())) < 25


def test_layout_preview_survives_a_scene_with_no_vehicles(tmp_path) -> None:
    image = Image.new("RGB", (320, 240), (10, 10, 10))
    destination = tmp_path / "empty.jpg"
    _overlay(image, [], destination, vehicles=[], candidates=[])
    assert destination.is_file()


def test_layout_correction_rejects_a_body_that_is_not_declared_as_json() -> None:
    """The exact shape of the reported "Request validation failed".

    `fetch` sends a string body as text/plain unless told otherwise, and FastAPI
    then hands Pydantic the raw string. The frontend client now declares the
    content type; this pins the server behaviour that made the omission fatal,
    so the two halves of the contract cannot drift apart silently again.
    """
    from fastapi.testclient import TestClient

    from app.main import app

    payload = '{"slots": [{"polygon": [[0.1,0.1],[0.2,0.1],[0.2,0.2],[0.1,0.2]]}]}'
    with TestClient(app) as client:
        untyped = client.post(
            "/api/v1/media/layouts/1/corrections",
            content=payload,
            headers={"Content-Type": "text/plain"},
        )
        assert untyped.status_code == 422
        assert any(
            item["type"] == "model_attributes_type"
            for item in untyped.json()["error"]["details"]
        )
        # Declared as JSON the same bytes get as far as the layout lookup,
        # i.e. past validation. 404 (no such layout) is the pass condition here.
        typed = client.post(
            "/api/v1/media/layouts/999999/corrections",
            content=payload,
            headers={"Content-Type": "application/json"},
        )
        assert typed.status_code == 404


# --- parking-area reasoning ------------------------------------------------
#
# The failure these guard against is circular: a false vehicle detection seeds
# parking surface, the surface then contains the detection, and the detection is
# reported as "inside the parking area" on the strength of a region it created
# itself. Vehicles may extend bay-anchored surface; they may not invent it.

def bay_row(count: int, *, y: float, x0: float = 0.10, width: float = 0.05,
            height: float = 0.09, gap: float = 0.012) -> list[list[list[float]]]:
    """A row of rectangular bays, in normalised coordinates."""
    bays = []
    for index in range(count):
        left = x0 + index * (width + gap)
        bays.append([[left, y], [left + width, y], [left + width, y + height], [left, y + height]])
    return bays


def test_an_isolated_false_detection_cannot_validate_itself() -> None:
    from app.ml.parking_area import (
        OUTSIDE_PARKING_AREA,
        classify_vehicle_positions,
        infer_parking_area,
    )

    bays = bay_row(8, y=0.20)
    # A lone detection far from every bay -- a lorry on a road, or a phantom.
    stray = (0.86, 0.86, 0.95, 0.95)
    area = infer_parking_area(bays, [stray])
    placements = classify_vehicle_positions(area, [stray], set())
    assert placements == [OUTSIDE_PARKING_AREA], (
        "an isolated detection grew a region around itself and was then found "
        "inside it -- the self-validating loop is open again"
    )


def test_a_road_beside_the_lot_stays_outside_it() -> None:
    from app.ml.parking_area import (
        OUTSIDE_PARKING_AREA,
        classify_vehicle_positions,
        infer_parking_area,
    )

    bays = bay_row(8, y=0.15)
    # Traffic well below the bays, across a kerb.
    road = [(0.10 + i * 0.09, 0.86, 0.17 + i * 0.09, 0.93) for i in range(6)]
    area = infer_parking_area(bays, road)
    placements = classify_vehicle_positions(area, road, set())
    assert all(placement == OUTSIDE_PARKING_AREA for placement in placements), placements


def test_unlabelled_rows_next_to_labelled_ones_stay_inside_the_facility() -> None:
    """The case vehicle seeding exists for, and must survive the anchoring rule.

    CNRPark camera 1 annotates two rows of a five-row car park; the cars in the
    unlabelled rows are genuinely parked on the site and must not be exiled.
    """
    from app.ml.parking_area import (
        IN_PARKING_AREA,
        classify_vehicle_positions,
        infer_parking_area,
    )

    bays = bay_row(8, y=0.20)
    # A further row of parked cars just beyond the labelled bays.
    unlabelled = [(0.10 + i * 0.062, 0.33, 0.15 + i * 0.062, 0.41) for i in range(8)]
    area = infer_parking_area(bays, unlabelled)
    placements = classify_vehicle_positions(area, unlabelled, set())
    assert placements.count(IN_PARKING_AREA) >= 6, placements


def test_a_partial_layout_still_yields_a_usable_area() -> None:
    from app.ml.parking_area import infer_parking_area

    assert infer_parking_area(bay_row(4, y=0.40), []).known


def test_no_bays_means_no_area_rather_than_a_guess() -> None:
    from app.ml.parking_area import AREA_UNKNOWN, classify_vehicle_positions, infer_parking_area

    area = infer_parking_area([], [(0.2, 0.2, 0.3, 0.3)])
    assert not area.known
    assert classify_vehicle_positions(area, [(0.2, 0.2, 0.3, 0.3)], set()) == [AREA_UNKNOWN]


def test_a_mapped_vehicle_is_never_reclassified_by_the_area() -> None:
    """Association wins: a vehicle in a detected bay is in a space, full stop."""
    from app.ml.parking_area import (
        IN_MAPPED_SPACE,
        classify_vehicle_positions,
        infer_parking_area,
    )

    bays = bay_row(6, y=0.20)
    boxes = [(0.11, 0.20, 0.15, 0.29), (0.95, 0.95, 0.99, 0.99)]
    area = infer_parking_area(bays, boxes)
    assert classify_vehicle_positions(area, boxes, {0})[0] == IN_MAPPED_SPACE
