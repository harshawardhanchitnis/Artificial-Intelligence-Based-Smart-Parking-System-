"""Regression tests for every defect fixed in the V3 work.

Each test names the defect it locks down so a future change that reintroduces
it fails here rather than in front of a user.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from app.datasets.labels import parse_occupancy_attribute
from app.datasets.obb_dataset import (
    KNOWN_HOLDOUT_FRACTION,
    assign_split,
    camera_identity,
    polygon_to_obb,
)
from app.datasets.v2_protocol import MAX_UNLABELLED_SLOT_RATE, _pklot_slots
from app.ml import registry
from app.ml.camera_fingerprint import camera_signature, fingerprint_distance, same_camera
from app.ml.generalized_localizer import letterbox, polygon_nms, rotated_box_corners
from app.ml.geometry import rectify_slot, rectify_slots
from app.services.localization_service import (
    AUTO_DETECTED,
    VERIFICATION_REQUIRED,
    LayoutResolution,
    layout_agreement,
)

# --------------------------------------------------------------------------
# P0-3 / E-5 -- a missing PKLot occupancy attribute must never mean "vacant"
# --------------------------------------------------------------------------

_SPACE_WITH_LABEL = """<parking id="test">
  <space id="1" occupied="1">
    <contour>
      <point x="10" y="10" /><point x="50" y="10" />
      <point x="50" y="40" /><point x="10" y="40" />
    </contour>
  </space>
</parking>"""

_SPACE_WITHOUT_LABEL = """<parking id="test">
  <space id="1">
    <contour>
      <point x="10" y="10" /><point x="50" y="10" />
      <point x="50" y="40" /><point x="10" y="40" />
    </contour>
  </space>
</parking>"""


def test_missing_occupied_attribute_is_unlabelled_not_vacant() -> None:
    assert parse_occupancy_attribute("1") is True
    assert parse_occupancy_attribute("0") is False
    # The defect: these all used to evaluate to False, i.e. a confident "vacant".
    assert parse_occupancy_attribute(None) is None
    assert parse_occupancy_attribute("") is None
    assert parse_occupancy_attribute("unknown") is None


def test_pklot_parser_drops_and_counts_unlabelled_spaces() -> None:
    labelled, unlabelled = _pklot_slots(_SPACE_WITH_LABEL.encode(), 100, 100)
    assert len(labelled) == 1
    assert labelled[0]["occupied"] is True
    assert unlabelled == 0

    dropped, missing = _pklot_slots(_SPACE_WITHOUT_LABEL.encode(), 100, 100)
    assert dropped == []
    assert missing == 1


def test_unlabelled_rate_threshold_quarantines_a_corrupt_source() -> None:
    # The three affected PUCPR dates carry roughly 98% unlabelled spaces.
    assert 0.98 > MAX_UNLABELLED_SLOT_RATE
    assert MAX_UNLABELLED_SLOT_RATE < 0.5


# --------------------------------------------------------------------------
# P0-1 / E-1 -- an unrecognised camera must not receive stored geometry
# --------------------------------------------------------------------------


def test_layout_agreement_survives_a_detector_with_low_recall() -> None:
    """A correct layout must still be trusted when the detector finds only some of it.

    Measured case: a valid UFPR05 video reference frame where the detector found
    13 of 40 spaces.  Scoring recalled-space coverage rejected it at 0.225;
    scoring the share of detections explained accepts it at 0.692.
    """
    recalled = [
        {"polygon": [[0.05 * i, 0.10], [0.05 * i + 0.04, 0.10],
                     [0.05 * i + 0.04, 0.18], [0.05 * i, 0.18]]}
        for i in range(1, 16)
    ]
    # The detector found only five of them, but every one lands on a real space.
    detected = recalled[:5]
    assert layout_agreement(recalled, detected) == 1.0


def test_layout_agreement_needs_enough_detections_to_mean_anything() -> None:
    recalled = [{"polygon": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]]}]
    assert layout_agreement(recalled, recalled) == 0.0


def test_layout_agreement_rejects_geometry_from_a_different_lot() -> None:
    recalled = [
        {"polygon": [[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]]},
        {"polygon": [[0.3, 0.1], [0.4, 0.1], [0.4, 0.2], [0.3, 0.2]]},
    ]
    recalled = [
        {"polygon": [[0.05 * i, 0.10], [0.05 * i + 0.04, 0.10],
                     [0.05 * i + 0.04, 0.18], [0.05 * i, 0.18]]}
        for i in range(1, 12)
    ]
    matching = list(recalled)
    elsewhere = [
        {"polygon": [[0.70, 0.05 * i], [0.80, 0.05 * i],
                     [0.80, 0.05 * i + 0.04], [0.70, 0.05 * i + 0.04]]}
        for i in range(1, 12)
    ]

    assert layout_agreement(recalled, matching) == 1.0
    assert layout_agreement(recalled, elsewhere) == 0.0
    # No independent detection means no corroboration, so no automatic result.
    assert layout_agreement(recalled, []) == 0.0


def test_resolution_without_verification_is_not_usable() -> None:
    blocked = LayoutResolution(
        slots=[{"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
        state=VERIFICATION_REQUIRED,
    )
    assert blocked.usable is False
    assert blocked.as_dict()["status"] == "verification_required"

    allowed = LayoutResolution(
        slots=[{"polygon": [[0, 0], [1, 0], [1, 1], [0, 1]]}],
        state=AUTO_DETECTED,
    )
    assert allowed.usable is True
    assert allowed.as_dict()["status"] == "success"


def test_a_resolution_with_no_slots_is_never_usable() -> None:
    assert LayoutResolution(slots=[], state=AUTO_DETECTED).usable is False


# --------------------------------------------------------------------------
# Camera fingerprinting -- a wrong match would reinstate the P0 defect
# --------------------------------------------------------------------------


def _noise_image(seed: int, size: tuple[int, int] = (640, 480)) -> Image.Image:
    generator = np.random.default_rng(seed)
    return Image.fromarray(
        generator.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8), mode="RGB"
    )


def test_identical_scenes_match_and_unrelated_scenes_do_not() -> None:
    first = _noise_image(1)
    signature = camera_signature(first)
    assert same_camera(signature, signature.fingerprint, signature.width, signature.height)

    other = camera_signature(_noise_image(99))
    assert not same_camera(
        other, signature.fingerprint, signature.width, signature.height, stored_signature=signature
    )


def test_aspect_ratio_alone_can_rule_out_a_match() -> None:
    signature = camera_signature(_noise_image(5, (1280, 720)), with_structure=False)
    # Same hash, incompatible camera geometry: still not the same camera.
    assert not same_camera(signature, signature.fingerprint, 800, 800)


def test_fingerprint_distance_is_defensive_about_bad_input() -> None:
    assert fingerprint_distance("not-a-hash", "also-bad") == 64


# --------------------------------------------------------------------------
# P0-4 / L-1 -- the rectification rewrite must be behaviour preserving
# --------------------------------------------------------------------------


def test_batch_and_single_rectification_agree() -> None:
    generator = np.random.default_rng(3)
    image = Image.fromarray(generator.integers(0, 255, (400, 600, 3), dtype=np.uint8), "RGB")
    polygons = [
        [[0.10, 0.10], [0.30, 0.12], [0.29, 0.40], [0.09, 0.38]],
        [[0.55, 0.50], [0.80, 0.52], [0.79, 0.88], [0.54, 0.86]],
    ]
    batched = rectify_slots(image, polygons)
    for polygon, patch in zip(polygons, batched, strict=True):
        single = rectify_slot(image, polygon)
        assert np.array_equal(np.asarray(single), np.asarray(patch))
        assert patch.size == (128, 128)


def test_rectification_rejects_an_invalid_polygon() -> None:
    image = Image.new("RGB", (200, 200))
    with pytest.raises(ValueError, match="Invalid parking-space polygon"):
        rectify_slots(image, [[[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]])


# --------------------------------------------------------------------------
# P1-1 -- predictors and metadata must be cached, not rebuilt per request
# --------------------------------------------------------------------------


def test_registry_rebuilds_only_when_the_file_signature_changes() -> None:
    registry.clear()
    calls = {"count": 0}

    def build() -> object:
        calls["count"] += 1
        return object()

    first = registry.cached("probe", ("v1",), build)
    second = registry.cached("probe", ("v1",), build)
    assert first is second
    assert calls["count"] == 1

    third = registry.cached("probe", ("v2",), build)
    assert third is not first
    assert calls["count"] == 2
    registry.clear()


def test_inference_providers_always_offer_a_cpu_path(monkeypatch) -> None:
    monkeypatch.delenv("PARKING_ONNX_PROVIDER", raising=False)
    assert registry.inference_providers() == ["CPUExecutionProvider"]
    monkeypatch.setenv("PARKING_ONNX_PROVIDER", "cuda")
    assert registry.inference_providers()[-1] == "CPUExecutionProvider"


# --------------------------------------------------------------------------
# P3-1 -- the generalized detector's inference contract
# --------------------------------------------------------------------------


def test_letterbox_preserves_aspect_ratio_and_reports_padding() -> None:
    canvas, scale, pad_x, pad_y = letterbox(Image.new("RGB", (1280, 720)), 1024)
    assert canvas.shape == (1024, 1024, 3)
    assert scale == pytest.approx(0.8)
    assert pad_x == 0.0
    assert pad_y == pytest.approx((1024 - 576) / 2)


def test_rotated_box_corners_are_ordered_and_sized() -> None:
    corners = rotated_box_corners(100.0, 100.0, 40.0, 20.0, 0.0)
    assert corners == [[80.0, 90.0], [120.0, 90.0], [120.0, 110.0], [80.0, 110.0]]


def test_polygon_nms_suppresses_overlapping_detections() -> None:
    box = [[0.10, 0.10], [0.20, 0.10], [0.20, 0.20], [0.10, 0.20]]
    nudged = [[0.105, 0.105], [0.205, 0.105], [0.205, 0.205], [0.105, 0.205]]
    far = [[0.70, 0.70], [0.80, 0.70], [0.80, 0.80], [0.70, 0.80]]
    detections = [
        {"polygon": box, "confidence": 0.9},
        {"polygon": nudged, "confidence": 0.8},
        {"polygon": far, "confidence": 0.7},
    ]
    kept = polygon_nms(detections, 0.30, 100)
    assert len(kept) == 2
    assert kept[0]["confidence"] == 0.9


# --------------------------------------------------------------------------
# P3-1 -- the leave-camera-out split must not leak
# --------------------------------------------------------------------------


def test_split_assignment_keeps_unseen_cameras_out_of_training() -> None:
    unseen = {"localization_partition": "validation", "dataset": "PKLot"}
    held = {"localization_partition": "holdout", "dataset": "PKLot"}
    assert assign_split(unseen) == "val_unseen"
    assert assign_split(held) == "test_unseen"


def test_acpds_never_enters_the_known_camera_split() -> None:
    # Every ACPDS capture is its own viewpoint, so holding one out would make it
    # an unseen camera masquerading as a known one.
    for index in range(25):
        row = {
            "localization_partition": "train",
            "dataset": "ACPDS",
            "source_id": f"train/GOPR{index:04d}",
        }
        assert assign_split(row) == "train"


def test_known_holdout_reserves_some_frames_from_multi_image_cameras() -> None:
    rows = [
        {"localization_partition": "train", "dataset": "PKLot", "source_id": f"PUCPR/x/{i}"}
        for i in range(400)
    ]
    held = sum(assign_split(row) == "val_known" for row in rows)
    assert 0 < held < len(rows)
    assert abs(held / len(rows) - KNOWN_HOLDOUT_FRACTION) < 0.08


def test_camera_identity_separates_sites_and_scenes() -> None:
    assert camera_identity({"dataset": "PKLot", "group_id": "PUCPR/2012-09-12"}) == "PKLot:PUCPR"
    assert (
        camera_identity({"dataset": "CNRPark+EXT", "group_id": "camera7/2016-01-14"})
        == "CNRPark+EXT:camera7"
    )
    assert (
        camera_identity({"dataset": "ACPDS", "source_id": "test/GOPR6543"})
        == "ACPDS:test/GOPR6543"
    )


def test_obb_conversion_rejects_unusable_polygons() -> None:
    good = polygon_to_obb([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]])
    assert good is not None
    assert len(good) == 8
    assert polygon_to_obb([[0.1, 0.1], [0.2, 0.1], [0.2, 0.2]]) is None
    assert polygon_to_obb([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1], [0.1, 0.1]]) is None
    assert polygon_to_obb([[-0.1, 0.1], [0.2, 0.1], [0.2, 0.2], [0.1, 0.2]]) is None


# --------------------------------------------------------------------------
# P1-5 / E-4 -- unlabelled predictions must not be scored as vacant
# --------------------------------------------------------------------------


def test_diagnostics_excludes_predictions_without_ground_truth() -> None:
    from app.services.diagnostics_service import application_run_diagnostics

    class Record:
        def __init__(self, identifier: int, predictions: list[dict[str, object]]) -> None:
            self.id = identifier
            self.dataset = "User upload"
            self.scenario_id = f"upload:{identifier}"
            self.prediction_json = json.dumps(predictions)

    labelled = Record(
        1, [{"id": "1", "predicted_occupied": True, "ground_truth_occupied": True}]
    )
    unlabelled = Record(2, [{"id": "1", "predicted_occupied": True}])

    report = application_run_diagnostics([labelled, unlabelled])
    # The unlabelled prediction used to be counted as a false-occupied error.
    assert report["unlabelled_predictions_excluded"] == 1
    assert report["unlabelled_runs_excluded"] == 1
    assert report["prediction_enabled_runs"] == 1


# --------------------------------------------------------------------------
# Readiness must surface a missing generalized detector
# --------------------------------------------------------------------------


def test_readiness_reports_a_missing_generalized_detector(tmp_path, monkeypatch) -> None:
    """A missing detector degrades the product to verification-only, so it is
    reported rather than passing silently."""
    from sqlalchemy import create_engine

    from app.services import reliability_service

    monkeypatch.setattr(
        reliability_service,
        "model_status",
        lambda _root: {
            "ready": True,
            "model_name": "occupancy",
            "independent_benchmark": {"unseen_test": {"unique_samples": 10}},
            "enhanced_occupancy": {"ready": True, "model_name": "enhanced"},
            "slot_localizer": {"ready": True, "model_name": "localizer"},
            "space_detector": {"ready": False, "reason": "not installed"},
        },
    )
    report = reliability_service.collect_readiness(
        tmp_path, tmp_path / "models", create_engine("sqlite://")
    )
    detector = next(c for c in report["checks"] if c["key"] == "space_detector")
    assert detector["ready"] is False
    assert report["ready"] is False
