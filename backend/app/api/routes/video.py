from __future__ import annotations

import json
from pathlib import Path

import cv2
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import AnalysisJob, AnalysisRecord, MediaAsset, OccupancyEvent, VideoAnalysis
from app.db.session import SessionLocal
from app.services.media_service import (
    MediaValidationError,
    resolve_media_path,
    safe_display_name,
    store_video,
)
from app.services.video_service import submit_video_job

router = APIRouter()


def _job_payload(job: AnalysisJob) -> dict[str, object]:
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "phase": job.phase,
        "progress": round(job.progress, 4),
        "cancel_requested": job.cancel_requested,
        "error": (
            {"code": job.error_code, "message": job.error_message} if job.error_code else None
        ),
        "result_analysis_id": job.result_analysis_id,
        "created_at": job.created_at,
        "started_at": job.started_at,
        "finished_at": job.finished_at,
    }


def _queue_job(session: Session, media: MediaAsset) -> dict[str, object]:
    job = AnalysisJob(media_asset_id=media.id, kind="fixed_camera_video")
    session.add(job)
    session.commit()
    session.refresh(job)
    payload = _job_payload(job)
    submit_video_job(job.id)
    return {**payload, "poll_url": f"/api/v1/video/jobs/{job.id}"}


@router.post("/analyse", status_code=202)
async def analyse_video(
    file: UploadFile = File(...),  # noqa: B008
) -> dict[str, object]:
    settings = get_settings()
    try:
        stored = await store_video(file, settings)
    except MediaValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    with SessionLocal() as session:
        media = MediaAsset(
            media_type="video",
            original_name=safe_display_name(file.filename, "parking-video.mp4"),
            storage_path=stored.relative_path,
            sha256=stored.sha256,
            mime_type=stored.mime_type,
            size_bytes=stored.size_bytes,
            width=stored.width,
            height=stored.height,
            duration_seconds=stored.duration_seconds,
            fps=stored.fps,
            frame_count=stored.frame_count,
        )
        existing = session.scalar(
            select(MediaAsset).where(
                MediaAsset.media_type == "video", MediaAsset.sha256 == stored.sha256
            )
        )
        if existing is not None:
            resolve_media_path(settings, stored.relative_path).unlink(missing_ok=True)
            media = existing
        else:
            session.add(media)
            session.flush()
        return _queue_job(session, media)


@router.post("/prepared/{video_id}/analyse", status_code=202)
def analyse_prepared_video(video_id: str) -> dict[str, object]:
    settings = get_settings()
    catalogue_path = settings.parking_data_root / "demo" / "videos" / "catalogue.json"
    if not catalogue_path.is_file():
        raise HTTPException(status_code=404, detail="Prepared video catalogue not found")
    catalogue = json.loads(catalogue_path.read_text(encoding="utf-8"))
    row = next(
        (value for value in catalogue.get("videos", []) if value.get("id") == video_id), None
    )
    if row is None:
        raise HTTPException(status_code=404, detail="Prepared video not found")
    path = resolve_media_path(settings, str(row["video_path"]))
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise HTTPException(status_code=422, detail="Prepared video cannot be decoded")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    capture.release()
    with SessionLocal() as session:
        media = session.scalar(
            select(MediaAsset).where(
                MediaAsset.media_type == "video", MediaAsset.sha256 == str(row["sha256"])
            )
        )
        if media is None:
            media = MediaAsset(
                media_type="video",
                original_name=Path(str(row["video_path"])).name,
                storage_path=str(row["video_path"]),
                sha256=str(row["sha256"]),
                mime_type="video/mp4",
                size_bytes=path.stat().st_size,
                width=width,
                height=height,
                duration_seconds=float(row["duration_seconds"]),
                fps=float(row["fps"]),
                frame_count=int(row["frame_count"]),
            )
            session.add(media)
            session.flush()
        return _queue_job(session, media)


@router.get("/jobs/{job_id}")
def video_job(job_id: int) -> dict[str, object]:
    with SessionLocal() as session:
        job = session.get(AnalysisJob, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Video job not found")
        return _job_payload(job)


@router.post("/jobs/{job_id}/cancel")
def cancel_video_job(job_id: int) -> dict[str, object]:
    with SessionLocal() as session:
        job = session.get(AnalysisJob, job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Video job not found")
        if job.status in {"completed", "failed", "cancelled", "interrupted"}:
            return _job_payload(job)
        job.cancel_requested = True
        job.phase = "cancellation_requested"
        session.commit()
        session.refresh(job)
        return _job_payload(job)


@router.get("/analyses/{analysis_id}")
def video_result(analysis_id: int) -> dict[str, object]:
    with SessionLocal() as session:
        result = session.scalar(
            select(VideoAnalysis).where(VideoAnalysis.analysis_id == analysis_id)
        )
        if result is None:
            raise HTTPException(status_code=404, detail="Video analysis not found")
        analysis = session.get(AnalysisRecord, analysis_id)
        media = session.get(MediaAsset, result.media_asset_id)
        events = session.scalars(
            select(OccupancyEvent)
            .where(OccupancyEvent.video_analysis_id == result.id)
            .order_by(OccupancyEvent.timestamp_seconds)
        ).all()
        return {
            "analysis_id": analysis_id,
            "sample_fps": result.sample_fps,
            "processed_frames": result.processed_frames,
            "dropped_frames": result.dropped_frames,
            "stability_confidence": result.stability_confidence,
            "processing_time_ms": analysis.processing_time_ms if analysis else None,
            "analysed_frames_per_second": (
                round(result.processed_frames / (analysis.processing_time_ms / 1_000), 3)
                if analysis and analysis.processing_time_ms > 0
                else None
            ),
            "source_duration_seconds": media.duration_seconds if media else None,
            "timeline": json.loads(result.timeline_json),
            "events": [
                {
                    "slot_index": event.slot_index,
                    "timestamp_seconds": event.timestamp_seconds,
                    "previous_state": event.previous_state,
                    "next_state": event.next_state,
                    "confidence": event.confidence,
                }
                for event in events
            ],
            "playback_url": f"/api/v1/video/analyses/{analysis_id}/playback",
        }


@router.get("/analyses/{analysis_id}/playback")
def video_playback(analysis_id: int) -> FileResponse:
    settings = get_settings()
    with SessionLocal() as session:
        result = session.scalar(
            select(VideoAnalysis).where(VideoAnalysis.analysis_id == analysis_id)
        )
        if result is None or not result.result_video_path:
            raise HTTPException(status_code=404, detail="Result video not found")
        path = resolve_media_path(settings, result.result_video_path)
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Result video file not found")
    return FileResponse(
        path,
        media_type="video/mp4",
        filename=f"parking-video-{analysis_id}.mp4",
        content_disposition_type="inline",
    )
