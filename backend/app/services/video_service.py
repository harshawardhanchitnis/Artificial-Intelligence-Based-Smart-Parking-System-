from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from PIL import Image
from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import (
    AnalysisJob,
    AnalysisRecord,
    DetectedLayout,
    DetectedSlot,
    MediaAsset,
    OccupancyEvent,
    VideoAnalysis,
)
from app.db.session import SessionLocal
from app.ml.association import AssociationResult, VehicleLink, associate
from app.ml.geometry import rectify_slots
from app.ml.occupancy_fusion import OccupancyEvidence, load_policy
from app.ml.occupancy_v3 import OccupancyV3Predictor
from app.ml.parking_area import (
    IN_PARKING_AREA,
    OUTSIDE_PARKING_AREA,
    ParkingArea,
    classify_vehicle_positions,
    infer_parking_area,
)
from app.ml.temporal import TemporalState, load_temporal_parameters
from app.ml.vehicle_detector import VehicleDetection, vehicle_counts
from app.ml.vehicle_tracking import VehicleTracker
from app.services.auto_calibration import CalibrationResult, calibrate_and_store
from app.services.localization_service import (
    AUTO_DETECTED,
    LayoutResolution,
    resolve_layout,
)
from app.services.media_service import resolve_media_path
from app.services.scene_analysis import detect_vehicles

_LOGGER = logging.getLogger(__name__)

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parking-video")
_SUBMIT_LOCK = threading.Lock()


class LayoutVerificationRequired(RuntimeError):
    """Raised when a video's parking-space geometry could not be established."""


class VideoEncodingError(RuntimeError):
    """Raised when the processed video cannot be encoded for playback.

    Carries a user-facing message; the encoder's own diagnostics stay in
    ``detail`` for the log rather than being shown to an operator.
    """

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.detail = detail


def resolve_ffmpeg() -> str | None:
    """Locate the FFmpeg binary, preferring explicit configuration.

    Order: the ``FFMPEG_PATH`` setting, then PATH, then a conventional
    per-user install directory.  Returning ``None`` lets readiness report the
    problem before a job spends minutes processing frames it cannot encode.
    """
    configured = get_settings().ffmpeg_path.strip()
    if configured:
        candidate = Path(configured)
        if candidate.is_file():
            return str(candidate)
        located = shutil.which(configured)
        if located:
            return located
    found = shutil.which("ffmpeg")
    if found:
        return found
    for root in (Path.home() / "Tools" / "ffmpeg",):
        matches = sorted(root.glob("**/bin/ffmpeg.exe")) + sorted(root.glob("**/bin/ffmpeg"))
        if matches:
            return str(matches[0])
    return None


def _encode_browser_video(source_path: Path, output_path: Path) -> None:
    ffmpeg = resolve_ffmpeg()

    if ffmpeg is None:
        raise VideoEncodingError(
            "Video encoding is unavailable because FFmpeg is not installed. "
            "Install FFmpeg or set FFMPEG_PATH, then submit the video again.",
            "resolve_ffmpeg() found no binary on PATH or in the configured location",
        )

    output_path.unlink(missing_ok=True)

    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source_path),
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(output_path),
    ]

    completed = subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )

    if completed.returncode != 0:
        output_path.unlink(missing_ok=True)
        detail = (completed.stderr or completed.stdout or "Unknown FFmpeg error").strip()
        raise VideoEncodingError(
            "The processed video could not be encoded for playback.",
            detail[-2000:],
        )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise VideoEncodingError(
            "The processed video could not be encoded for playback.",
            "FFmpeg exited successfully but produced no output file",
        )


def recover_interrupted_jobs() -> int:
    with SessionLocal() as session:
        jobs = session.scalars(
            select(AnalysisJob).where(AnalysisJob.status.in_(["queued", "running"]))
        ).all()
        for job in jobs:
            job.status = "interrupted"
            job.phase = "restart_required"
            job.error_code = "PROCESS_RESTARTED"
            job.error_message = (
                "Processing was interrupted by an application restart. Submit again."
            )
            job.finished_at = datetime.now(UTC)
        session.commit()
        return len(jobs)


def submit_video_job(job_id: int) -> None:
    with _SUBMIT_LOCK:
        _EXECUTOR.submit(_process_video, job_id)


def _update_job(job_id: int, **values: object) -> bool:
    with SessionLocal() as session:
        job = session.get(AnalysisJob, job_id)
        if job is None:
            return False
        if job.cancel_requested and values.get("status") not in {"cancelled", "failed"}:
            return False
        for key, value in values.items():
            setattr(job, key, value)
        session.commit()
        return True


def _cancelled(job_id: int) -> bool:
    with SessionLocal() as session:
        job = session.get(AnalysisJob, job_id)
        return bool(job is None or job.cancel_requested)


def _reference_frame(capture: cv2.VideoCapture, fps: float) -> np.ndarray:
    frames = []
    for second in (0.0, 0.75, 1.5, 2.25, 3.0):
        capture.set(cv2.CAP_PROP_POS_MSEC, second * 1_000)
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(frame)
    if not frames:
        raise RuntimeError("No reference frame could be decoded")
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return np.median(np.stack(frames), axis=0).astype(np.uint8)


class CameraStabilityTracker:
    """Measures how well each frame still matches the job's reference frame.

    The reference never changes during a job, so its ORB detector, keypoints and
    descriptors are computed once here instead of once per frame.
    """

    def __init__(self, reference: np.ndarray) -> None:
        self._detector = cv2.ORB_create(nfeatures=500)
        self._matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self._keypoints, self._descriptors = self._detector.detectAndCompute(
            cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None
        )

    def score(self, frame: np.ndarray) -> float:
        if self._descriptors is None:
            return 0.0
        keypoints, descriptors = self._detector.detectAndCompute(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None
        )
        if descriptors is None:
            return 0.0
        matches = self._matcher.match(self._descriptors, descriptors)
        if len(matches) < 12:
            return 0.0
        matches = sorted(matches, key=lambda match: match.distance)[:100]
        source = np.float32([self._keypoints[match.queryIdx].pt for match in matches])
        target = np.float32([keypoints[match.trainIdx].pt for match in matches])
        _, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
        return float(np.mean(inliers)) if inliers is not None else 0.0


# Bay colours in BGR.  Amber is a third answer, not a weaker red.
_FRAME_COLOURS = {
    "occupied": (45, 45, 220),
    "vacant": (55, 170, 65),
    "uncertain": (6, 119, 217),
}
# Vehicles are outlined thinly so the parking map stays the dominant reading.
_FRAME_VEHICLE = (246, 130, 59)
_FRAME_UNMAPPED = (234, 51, 147)


def _draw_frame(
    frame: np.ndarray,
    predictions: list[dict[str, object]],
    *,
    vehicles: list[tuple[float, float, float, float]] | None = None,
    unmapped: set[int] | None = None,
) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    overlay = output.copy()
    unmapped = unmapped or set()
    for index, box in enumerate(vehicles or []):
        x1, y1, x2, y2 = box
        colour = _FRAME_UNMAPPED if index in unmapped else _FRAME_VEHICLE
        cv2.rectangle(
            output,
            (round(x1 * width), round(y1 * height)),
            (round(x2 * width), round(y2 * height)),
            colour,
            1,
        )
    for index, prediction in enumerate(predictions, 1):
        points = np.asarray(
            [[round(x * width), round(y * height)] for x, y in prediction["polygon"]],
            dtype=np.int32,
        )
        state = str(
            prediction.get("occupancy_state")
            or ("occupied" if prediction.get("predicted_occupied") else "vacant")
        )
        color = _FRAME_COLOURS.get(state, _FRAME_COLOURS["vacant"])
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(output, [points], True, color, 2)
        center = tuple(np.mean(points, axis=0).astype(int))
        cv2.putText(output, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.addWeighted(overlay, 0.22, output, 0.78, 0, output)
    return output


def _best_link(association: AssociationResult, slot_index: int) -> VehicleLink | None:
    """The strongest vehicle link for one bay, or ``None`` if nothing occupies it."""
    return max(
        association.occupying_links(slot_index),
        key=lambda link: link.slot_coverage,
        default=None,
    )


def _best_vehicle_confidence(
    association: AssociationResult, vehicles: list[VehicleDetection], slot_index: int
) -> float:
    link = _best_link(association, slot_index)
    return vehicles[link.vehicle_index].confidence if link is not None else 0.0


def _best_coverage(association: AssociationResult, slot_index: int, which: str) -> float:
    link = _best_link(association, slot_index)
    if link is None:
        return 0.0
    return link.slot_coverage if which == "slot" else link.vehicle_coverage


def _calibration_frames(
    capture: cv2.VideoCapture, source_fps: float, duration: float, count: int
) -> list[Image.Image]:
    """Sample frames spread across the clip for automatic layout calibration.

    Spread rather than consecutive: neighbouring frames of a fixed camera are
    nearly identical, so they would agree with each other about a spurious
    detection as readily as about a real bay and the consensus would prove
    nothing.  Frames seconds apart differ in traffic, light and shadow, so what
    survives all of them is the geometry rather than the contents.
    """
    span = max(duration, 1.0)
    frames: list[Image.Image] = []
    for index in range(count):
        position = span * (index + 0.5) / count
        capture.set(cv2.CAP_PROP_POS_MSEC, position * 1_000)
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return frames


def _process_video(job_id: int) -> None:
    settings = get_settings()
    started_clock = perf_counter()
    capture: cv2.VideoCapture | None = None
    writer: cv2.VideoWriter | None = None
    temporary_result_path: Path | None = None
    try:
        if not _update_job(
            job_id,
            status="running",
            phase="reference_layout",
            progress=0.02,
            started_at=datetime.now(UTC),
        ):
            return
        with SessionLocal() as session:
            job = session.get(AnalysisJob, job_id)
            media = session.get(MediaAsset, job.media_asset_id) if job else None
            if media is None:
                raise RuntimeError("Video media record is missing")
            media_id = media.id
            video_path = resolve_media_path(settings, media.storage_path)
            duration = float(media.duration_seconds or 0)
            source_fps = float(media.fps or 1)
            width, height = int(media.width or 0), int(media.height or 0)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError("Video decoder could not open the stored media")
        reference = _reference_frame(capture, source_fps)
        reference_image = Image.fromarray(cv2.cvtColor(reference, cv2.COLOR_BGR2RGB))
        with SessionLocal() as session:
            resolution = resolve_layout(session, settings.model_root, reference_image)
        calibration: CalibrationResult | None = None
        if not resolution.usable:
            # A fixed camera does not need a person to draw its bays.  Several
            # frames spread across the clip are detected independently and only
            # the geometry that recurs is kept, which removes the per-frame
            # accidents a single reference frame would have inherited.
            _update_job(job_id, phase="calibrating_camera", progress=0.04)
            frames = _calibration_frames(
                capture, source_fps, duration, settings.calibration_frames
            )
            with SessionLocal() as session:
                calibration, stored = calibrate_and_store(
                    session, settings, frames, f"Camera from job {job_id}"
                )
            if not calibration.established:
                # Still never guessed at, and old geometry is never stamped onto
                # unrelated footage -- but the failure is now the AI's own
                # verdict rather than a request for the operator to draw boxes.
                raise LayoutVerificationRequired(calibration.message)
            resolution = LayoutResolution(
                slots=calibration.slots,
                state=AUTO_DETECTED,
                source="automatic_calibration",
                confidence=calibration.consensus,
                fingerprint=stored.camera_fingerprint if stored else "",
                verified_layout_id=stored.id if stored else None,
                message=calibration.message,
                detector_available=True,
            )
        slots = resolution.slots
        with SessionLocal() as session:
            layout = DetectedLayout(
                media_asset_id=media_id,
                model_name=resolution.source,
                layout_identifier=uuid.uuid4().hex,
                confidence=float(resolution.confidence),
                status="automatic",
                verification_state=resolution.state,
                verified_layout_id=resolution.verified_layout_id,
                camera_fingerprint=resolution.fingerprint,
                reference_frame_seconds=0.0,
                geometry_json=json.dumps(slots, separators=(",", ":")),
            )
            session.add(layout)
            session.flush()
            for index, slot in enumerate(slots, 1):
                session.add(
                    DetectedSlot(
                        layout_id=layout.id,
                        slot_index=index,
                        polygon_json=json.dumps(slot["polygon"], separators=(",", ":")),
                        localization_confidence=float(
                            slot.get("localization_confidence", 1.0)
                        ),
                        corner_confidence=float(slot.get("corner_confidence", 1.0)),
                    )
                )
            session.commit()
            layout_id = layout.id
        predictor = OccupancyV3Predictor.load(settings.model_root)
        # Loaded once for the whole job rather than per frame.
        fusion = load_policy(settings.model_root)
        # Rejects detections that do not recur across frames, which is the
        # only evidence available against a one-frame artefact.
        vehicle_tracker = VehicleTracker()
        stability_tracker = CameraStabilityTracker(reference)
        slot_polygons = [slot["polygon"] for slot in slots]
        # The camera is fixed and the layout is established for the whole
        # job, so the facility's extent is derived once from the first frame
        # that supplies vehicles and reused for the rest.
        parking_area = ParkingArea()
        temporal = TemporalState(
            len(slots), predictor.threshold, load_temporal_parameters(settings.model_root)
        )
        sample_fps = min(settings.video_sample_fps, source_fps)
        sample_step = max(1, round(source_fps / sample_fps))
        expected = max(1, int(duration * sample_fps))
        relative_result = Path("media") / "results" / "videos" / f"{uuid.uuid4().hex}.mp4"
        result_path = settings.parking_data_root / relative_result
        temporary_result_path = result_path.with_name(
            f"{result_path.stem}.working.mp4"
        )
        result_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(temporary_result_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            sample_fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("MP4 result encoder is unavailable")
        timeline = []
        events = []
        processed = dropped = frame_index = 0
        stability_values = []
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % sample_step:
                frame_index += 1
                continue
            if _cancelled(job_id):
                capture.release()
                writer.release()
                temporary_result_path.unlink(missing_ok=True)
                _update_job(
                    job_id, status="cancelled", phase="cancelled", finished_at=datetime.now(UTC)
                )
                return
            timestamp = frame_index / source_fps
            stability = stability_tracker.score(frame)
            stability_values.append(stability)
            if stability < 0.35:
                dropped += 1
                timeline.append(
                    {
                        "timestamp_seconds": round(timestamp, 3),
                        "status": "uncertain_camera_motion",
                        "occupied_spaces": None,
                        "vacant_spaces": None,
                        "average_confidence": None,
                    }
                )
                frame_index += 1
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            patches = rectify_slots(image, slot_polygons)
            probabilities, _ = predictor.probabilities(patches)

            # Full-scene vehicle evidence for this frame.  The bay geometry is
            # already established and cached, so only this changes frame to
            # frame -- the space detector is not re-run.
            frame_vehicles, vehicle_available = detect_vehicles(image, settings)
            confirmed = vehicle_tracker.confirm([v.box for v in frame_vehicles])
            frame_vehicles = [
                vehicle
                for vehicle, ok in zip(frame_vehicles, confirmed, strict=True)
                if ok
            ]
            if not parking_area.known:
                parking_area = infer_parking_area(
                    slot_polygons, [vehicle.box for vehicle in frame_vehicles]
                )
            association = associate(
                [
                    [[x * width, y * height] for x, y in polygon]
                    for polygon in slot_polygons
                ],
                [
                    (
                        vehicle.box[0] * width,
                        vehicle.box[1] * height,
                        vehicle.box[2] * width,
                        vehicle.box[3] * height,
                    )
                    for vehicle in frame_vehicles
                ],
            )

            # Fuse first, then smooth.  Smoothing a fused probability keeps the
            # temporal machinery working on a single calibrated series; fusing
            # afterwards would apply vehicle evidence to a value that already
            # carries several frames of history and count it repeatedly.
            fused_probabilities = np.asarray(
                [
                    fusion.fuse(
                        OccupancyEvidence(
                            occupancy_probability=float(probability),
                            geometry_confidence=float(
                                slot.get("localization_confidence", 1.0)
                            ),
                            vehicle_evidence_available=vehicle_available,
                            vehicle_detected=bool(association.occupying_links(index)),
                            vehicle_confidence=_best_vehicle_confidence(
                                association, frame_vehicles, index
                            ),
                            slot_coverage=_best_coverage(association, index, "slot"),
                            vehicle_coverage=_best_coverage(association, index, "vehicle"),
                        )
                    ).probability
                    for index, (slot, probability) in enumerate(
                        zip(slots, probabilities, strict=True)
                    )
                ],
                dtype=np.float64,
            )
            smoothed, states, changes = temporal.update(fused_probabilities)
            vehicle_for_slot = {
                index: frame_vehicles[link.vehicle_index].product_class
                for index in range(len(slots))
                for link in association.occupying_links(index)[:1]
            }
            # One state assignment feeds every count and every colour, so the
            # three totals are mutually exclusive by construction.  The band
            # decides whether a verdict is asserted at all; the hysteresis state
            # decides which verdict it is, which keeps a bay from flickering
            # around the threshold.
            frame_states = [
                "uncertain"
                if predictor.state_for(float(probability)) == "uncertain"
                else ("occupied" if bool(state) else "vacant")
                for probability, state in zip(smoothed, states, strict=True)
            ]
            predictions = [
                {
                    **slot,
                    "predicted_occupied": frame_state == "occupied",
                    "occupancy_state": frame_state,
                    "occupied_probability": round(float(probability), 6),
                    "confidence": round(float(max(probability, 1 - probability)), 6),
                    "vehicle_class": vehicle_for_slot.get(index),
                }
                for index, (slot, probability, frame_state) in enumerate(
                    zip(slots, smoothed, frame_states, strict=True)
                )
            ]
            occupied = sum(1 for value in frame_states if value == "occupied")
            uncertain = sum(1 for value in frame_states if value == "uncertain")
            mapped_vehicles = {
                link.vehicle_index
                for index in range(len(slots))
                for link in association.occupying_links(index)
            }
            placements = classify_vehicle_positions(
                parking_area, [vehicle.box for vehicle in frame_vehicles], mapped_vehicles
            )
            facility_counts = vehicle_counts(
                [
                    vehicle
                    for vehicle, placement in zip(frame_vehicles, placements, strict=True)
                    if placement != OUTSIDE_PARKING_AREA
                ]
            )
            timeline.append(
                {
                    "timestamp_seconds": round(timestamp, 3),
                    "status": "observed",
                    "occupied_spaces": occupied,
                    "uncertain_spaces": uncertain,
                    # Uncertain spaces are withheld from both verdicts, so the
                    # three counts sum to the slot count.
                    "vacant_spaces": len(slots) - occupied - uncertain,
                    "average_confidence": round(
                        float(np.mean([prediction["confidence"] for prediction in predictions])), 6
                    ),
                    # Facility figures only: traffic behind the site is counted
                    # separately and never folded into the parking numbers.
                    "vehicle_counts": facility_counts,
                    "unmapped_vehicles": sum(
                        1 for placement in placements if placement == IN_PARKING_AREA
                    ),
                    "vehicles_outside_parking_area": sum(
                        1 for placement in placements if placement == OUTSIDE_PARKING_AREA
                    ),
                }
            )
            for slot_index, previous, next_state in changes:
                events.append(
                    {
                        "slot_index": slot_index + 1,
                        "timestamp_seconds": timestamp,
                        "previous_state": "occupied" if previous else "vacant",
                        "next_state": "occupied" if next_state else "vacant",
                        "confidence": float(max(smoothed[slot_index], 1 - smoothed[slot_index])),
                    }
                )
            writer.write(
                _draw_frame(
                    frame,
                    predictions,
                    vehicles=[vehicle.box for vehicle in frame_vehicles],
                    # Only vehicles on the site are drawn as unmapped; traffic
                    # behind the car park is drawn as an ordinary detection so
                    # the overlay does not imply it is a parking problem.
                    unmapped={
                        index
                        for index, placement in enumerate(placements)
                        if placement == IN_PARKING_AREA
                    },
                )
            )
            processed += 1
            if processed % 5 == 0:
                _update_job(
                    job_id,
                    phase="occupancy_tracking",
                    progress=min(0.97, 0.08 + 0.89 * processed / expected),
                )
            frame_index += 1
        capture.release()
        writer.release()
        if processed == 0:
            temporary_result_path.unlink(missing_ok=True)
            raise RuntimeError("No stable frames were available for fixed-camera analysis")

        if _cancelled(job_id):
            temporary_result_path.unlink(missing_ok=True)
            _update_job(
                job_id,
                status="cancelled",
                phase="cancelled",
                finished_at=datetime.now(UTC),
            )
            return

        _update_job(
            job_id,
            phase="browser_encoding",
            progress=0.985,
        )

        _encode_browser_video(temporary_result_path, result_path)
        temporary_result_path.unlink(missing_ok=True)

        final_values = next(
            value for value in reversed(timeline) if value.get("status") == "observed"
        )
        with SessionLocal() as session:
            record = AnalysisRecord(
                dataset="User video",
                scenario_id=f"video:{media_id}",
                total_spaces=len(slots),
                occupied_spaces=int(final_values["occupied_spaces"]),
                vacant_spaces=int(final_values["vacant_spaces"]),
                processing_time_ms=(perf_counter() - started_clock) * 1_000,
                result_image_path=None,
                model_name=str(predictor.metadata["model_name"]),
                average_confidence=float(final_values["average_confidence"]),
                prediction_json=None,
                source_type="video_upload",
                media_asset_id=media_id,
                job_id=job_id,
                layout_id=layout_id,
                localization_confidence=float(resolution.confidence),
                result_status=resolution.state,
            )
            session.add(record)
            session.flush()
            video = VideoAnalysis(
                analysis_id=record.id,
                media_asset_id=media_id,
                layout_id=layout_id,
                sample_fps=sample_fps,
                processed_frames=processed,
                dropped_frames=dropped,
                stability_confidence=float(np.mean(stability_values)),
                timeline_json=json.dumps(timeline, separators=(",", ":")),
                result_video_path=relative_result.as_posix(),
            )
            session.add(video)
            session.flush()
            for event in events:
                session.add(OccupancyEvent(video_analysis_id=video.id, **event))
            job = session.get(AnalysisJob, job_id)
            if job:
                job.status = "completed"
                job.phase = "completed"
                job.progress = 1.0
                job.result_analysis_id = record.id
                job.finished_at = datetime.now(UTC)
            session.commit()
    except LayoutVerificationRequired as exc:
        _update_job(
            job_id,
            status="failed",
            phase="layout_verification_required",
            error_code="LAYOUT_VERIFICATION_REQUIRED",
            error_message=str(exc)[:500],
            finished_at=datetime.now(UTC),
        )
    except Exception as exc:
        if isinstance(exc, VideoEncodingError):
            code, message, detail = "VIDEO_ENCODING_FAILED", str(exc), exc.detail
        else:
            code, message, detail = "VIDEO_PROCESSING_FAILED", str(exc), ""
        _LOGGER.exception("Video job %s failed: %s %s", job_id, code, detail)
        _update_job(
            job_id,
            status="failed",
            phase="failed",
            error_code=code,
            error_message=message[:500],
            finished_at=datetime.now(UTC),
        )
    finally:
        # Decoder and encoder handles and the OpenCV intermediate must not
        # survive a failure; the success path has already consumed them.
        if capture is not None:
            capture.release()
        if writer is not None:
            writer.release()
        if temporary_result_path is not None:
            temporary_result_path.unlink(missing_ok=True)
