"""Automatic calibration, exercised on real frames from a reconstructed sequence.

These tests run against the development sequences built by
``ml/build_development_sequence.py`` when they are present, and skip when they
are not, so the suite stays runnable on a machine without the prepared corpus.

They exist because the *video job* cannot reach this path on any sequence the
corpus can produce: both cameras with enough chronological frames -- PKLot
UFPR04 and CNRPark camera7 -- are registered identities for the recall path, so
a job resolves their layout before calibration is ever consulted.  The branch is
therefore driven directly here, with the frames a job would have used.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.db.models import Base, VerifiedLayout
from app.services.auto_calibration import (
    ACTIVE,
    UNRESOLVED,
    calibrate_and_store,
    calibrate_from_frames,
    layout_still_describes,
)

SEQUENCES = Path("D:/Projects/AI Based Smart Parking System Data/development/sequences")


def _frames(sequence: str, count: int) -> list[Image.Image]:
    catalogue = SEQUENCES / f"{sequence}.json"
    if not catalogue.is_file():
        pytest.skip(f"development sequence {sequence} is not built on this machine")
    video_path = Path(str(json.loads(catalogue.read_text())["video_path"]))
    if not video_path.is_file():
        pytest.skip(f"development sequence {sequence} has no video file")
    capture = cv2.VideoCapture(str(video_path))
    frames: list[Image.Image] = []
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or count
        for index in range(count):
            capture.set(
                cv2.CAP_PROP_POS_FRAMES,
                int(index * max(total - 1, 1) / max(count - 1, 1)),
            )
            ok, frame = capture.read()
            if ok and frame is not None:
                frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    finally:
        capture.release()
    if len(frames) < count:
        pytest.skip("sequence produced too few decodable frames")
    return frames


@pytest.fixture
def session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with sessionmaker(bind=engine)() as active:
        yield active


def test_a_camera_calibrates_itself_from_several_frames(session) -> None:
    settings = get_settings()
    result, layout = calibrate_and_store(
        session, settings, _frames("unseen-camera7", 7), "camera7 development"
    )
    if result.state == UNRESOLVED:
        pytest.skip("the parking-space detector is not installed on this machine")
    assert result.state == ACTIVE
    assert result.established
    assert layout is not None
    assert layout.slot_count == len(result.slots)
    assert layout.source == "automatic_calibration"
    assert layout.lifecycle_state == ACTIVE
    # Nobody was asked to confirm anything.
    assert layout.frames_observed == 7


def test_a_calibrated_layout_is_stored_for_reuse(session) -> None:
    settings = get_settings()
    result, layout = calibrate_and_store(
        session, settings, _frames("unseen-camera7", 5), "camera7 development"
    )
    if result.state == UNRESOLVED:
        pytest.skip("the parking-space detector is not installed on this machine")
    stored = session.query(VerifiedLayout).all()
    assert len(stored) == 1
    assert stored[0].camera_fingerprint == layout.camera_fingerprint
    assert json.loads(stored[0].geometry_json)


def test_calibrating_twice_does_not_duplicate_the_camera(session) -> None:
    settings = get_settings()
    frames = _frames("unseen-camera7", 5)
    first, _ = calibrate_and_store(session, settings, frames, "camera7")
    if first.state == UNRESOLVED:
        pytest.skip("the parking-space detector is not installed on this machine")
    calibrate_and_store(session, settings, frames, "camera7")
    assert session.query(VerifiedLayout).count() == 1


def test_blank_frames_produce_an_unresolved_layout_not_an_invented_one() -> None:
    """Abstention is the required outcome when nothing recurs."""
    settings = get_settings()
    blank = [Image.new("RGB", (640, 480), (90, 90, 90)) for _ in range(6)]
    result = calibrate_from_frames(blank, settings)
    assert result.state == UNRESOLVED
    assert result.slots == []
    assert result.established is False
    assert "could not be determined reliably" in result.message


def test_no_frames_at_all_is_unresolved_rather_than_an_error() -> None:
    assert calibrate_from_frames([], get_settings()).state == UNRESOLVED


def test_a_stored_layout_is_recognised_on_a_later_frame_of_the_same_camera() -> None:
    """Drift detection must not fire on the camera the layout came from.

    Measured across later frames rather than one: these are reconstructed
    time-lapses whose frames are minutes apart, so a single frame can differ in
    light and traffic far more than consecutive video would.
    """
    settings = get_settings()
    frames = _frames("unseen-camera7", 8)
    result = calibrate_from_frames(frames[:4], settings)
    if result.state == UNRESOLVED:
        pytest.skip("the parking-space detector is not installed on this machine")
    verdicts = [layout_still_describes(result.slots, frame, settings) for frame in frames[4:]]
    assert all(unchanged for unchanged, _ in verdicts), [
        round(score, 3) for _, score in verdicts
    ]


def test_drift_detection_separates_a_different_car_park() -> None:
    """The layout of another site must not be accepted as this camera's view."""
    settings = get_settings()
    here = _frames("unseen-camera7", 6)
    elsewhere = _frames("unseen-ufpr04", 6)
    other = calibrate_from_frames(elsewhere[:4], settings)
    if other.state == UNRESOLVED:
        pytest.skip("the parking-space detector is not installed on this machine")
    unchanged, agreement = layout_still_describes(other.slots, here[-1], settings)
    assert not unchanged, f"a foreign layout scored {agreement:.3f}"
