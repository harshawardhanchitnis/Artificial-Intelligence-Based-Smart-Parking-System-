from __future__ import annotations

import json
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
from app.ml.geometry import rectify_slot
from app.ml.localization import SlotLocalizerPredictor
from app.ml.occupancy_v3 import OccupancyV3Predictor
from app.ml.temporal import TemporalState, load_temporal_parameters
from app.services.media_service import resolve_media_path

_EXECUTOR = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parking-video")
_SUBMIT_LOCK = threading.Lock()


def _encode_browser_video(source_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")

    if ffmpeg is None:
        manual_root = Path.home() / "Tools" / "ffmpeg"
        candidates = sorted(manual_root.glob("**/bin/ffmpeg.exe"))
        if candidates:
            ffmpeg = str(candidates[0])

    if ffmpeg is None:
        raise RuntimeError(
            "FFmpeg was not found in PATH or under "
            f"{Path.home() / 'Tools' / 'ffmpeg'}."
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
        raise RuntimeError(
            f"Browser-compatible H.264 encoding failed: {detail[-800:]}"
        )

    if not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg did not produce a valid browser-compatible video")


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


def _camera_stability(reference: np.ndarray, frame: np.ndarray) -> float:
    detector = cv2.ORB_create(nfeatures=500)
    first_keypoints, first_descriptors = detector.detectAndCompute(
        cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), None
    )
    second_keypoints, second_descriptors = detector.detectAndCompute(
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None
    )
    if first_descriptors is None or second_descriptors is None:
        return 0.0
    matches = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True).match(
        first_descriptors, second_descriptors
    )
    if len(matches) < 12:
        return 0.0
    matches = sorted(matches, key=lambda match: match.distance)[:100]
    source = np.float32([first_keypoints[match.queryIdx].pt for match in matches])
    target = np.float32([second_keypoints[match.trainIdx].pt for match in matches])
    _, inliers = cv2.findHomography(source, target, cv2.RANSAC, 3.0)
    return float(np.mean(inliers)) if inliers is not None else 0.0


def _draw_frame(frame: np.ndarray, predictions: list[dict[str, object]]) -> np.ndarray:
    output = frame.copy()
    height, width = output.shape[:2]
    overlay = output.copy()
    for index, prediction in enumerate(predictions, 1):
        points = np.asarray(
            [[round(x * width), round(y * height)] for x, y in prediction["polygon"]],
            dtype=np.int32,
        )
        color = (45, 45, 220) if prediction["predicted_occupied"] else (55, 170, 65)
        cv2.fillPoly(overlay, [points], color)
        cv2.polylines(output, [points], True, color, 2)
        center = tuple(np.mean(points, axis=0).astype(int))
        cv2.putText(output, str(index), center, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    cv2.addWeighted(overlay, 0.22, output, 0.78, 0, output)
    return output


def _process_video(job_id: int) -> None:
    settings = get_settings()
    started_clock = perf_counter()
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
        localization = SlotLocalizerPredictor.load(settings.model_root).detect(reference_image)
        if localization["status"] == "unsupported_layout":
            raise RuntimeError(
                "Automatic localisation confidence is insufficient for this fixed-camera video"
            )
        slots = localization["slots"]
        with SessionLocal() as session:
            layout = DetectedLayout(
                media_asset_id=media_id,
                model_name=str(localization["model_name"]),
                layout_identifier=uuid.uuid4().hex,
                confidence=float(localization["confidence"]),
                status="automatic",
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
                        localization_confidence=float(slot["localization_confidence"]),
                        corner_confidence=float(slot["corner_confidence"]),
                    )
                )
            session.commit()
            layout_id = layout.id
        predictor = OccupancyV3Predictor.load(settings.model_root)
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
            stability = _camera_stability(reference, frame)
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
            patches = [rectify_slot(image, slot["polygon"]) for slot in slots]
            probabilities, _ = predictor.probabilities(patches)
            smoothed, states, changes = temporal.update(probabilities)
            predictions = [
                {
                    **slot,
                    "predicted_occupied": bool(state),
                    "occupied_probability": round(float(probability), 6),
                    "confidence": round(float(max(probability, 1 - probability)), 6),
                }
                for slot, probability, state in zip(slots, smoothed, states, strict=True)
            ]
            occupied = int(np.sum(states))
            timeline.append(
                {
                    "timestamp_seconds": round(timestamp, 3),
                    "status": "observed",
                    "occupied_spaces": occupied,
                    "vacant_spaces": len(slots) - occupied,
                    "average_confidence": round(
                        float(np.mean([prediction["confidence"] for prediction in predictions])), 6
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
            writer.write(_draw_frame(frame, predictions))
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
                localization_confidence=float(localization["confidence"]),
                result_status="success",
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
    except Exception as exc:
        _update_job(
            job_id,
            status="failed",
            phase="failed",
            error_code="VIDEO_PROCESSING_FAILED",
            error_message=str(exc)[:1000],
            finished_at=datetime.now(UTC),
        )
