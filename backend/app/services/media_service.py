from __future__ import annotations

import hashlib
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import cv2
from fastapi import UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError

from app.core.config import Settings


class MediaValidationError(ValueError):
    """Raised when uploaded media is unsafe or outside supported limits."""


@dataclass(frozen=True)
class StoredMedia:
    relative_path: str
    sha256: str
    size_bytes: int
    mime_type: str
    width: int
    height: int
    duration_seconds: float | None = None
    fps: float | None = None
    frame_count: int | None = None


def safe_display_name(filename: str | None, fallback: str) -> str:
    value = Path(filename or fallback).name
    value = re.sub(r"[^A-Za-z0-9._ -]+", "_", value).strip(" .")
    return value[:180] or fallback


async def _stream_upload(upload: UploadFile, root: Path, limit_bytes: int) -> tuple[Path, int, str]:
    root.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    suffix = Path(upload.filename or "upload.bin").suffix.lower()[:10]
    destination = root / f"{uuid.uuid4().hex}{suffix}"
    with tempfile.NamedTemporaryFile("w+b", dir=root, suffix=".upload", delete=False) as handle:
        temporary = Path(handle.name)
        while chunk := await upload.read(1024 * 1024):
            size += len(chunk)
            if size > limit_bytes:
                handle.close()
                temporary.unlink(missing_ok=True)
                raise MediaValidationError("Uploaded file exceeds the configured size limit")
            digest.update(chunk)
            handle.write(chunk)
    if size == 0:
        temporary.unlink(missing_ok=True)
        raise MediaValidationError("Uploaded file is empty")
    os.replace(temporary, destination)
    return destination, size, digest.hexdigest()


async def store_image(upload: UploadFile, settings: Settings) -> StoredMedia:
    path, size, digest = await _stream_upload(
        upload, settings.upload_root / "images", settings.max_image_upload_mb * 1024 * 1024
    )
    try:
        with Image.open(path) as candidate:
            candidate.verify()
        with Image.open(path) as candidate:
            image = ImageOps.exif_transpose(candidate).convert("RGB")
            width, height = image.size
            if max(width, height) > settings.max_image_dimension:
                raise MediaValidationError("Image dimensions exceed the configured limit")
            if min(width, height) < 160:
                raise MediaValidationError(
                    "Image is too small for reliable parking-space localisation"
                )
            canonical = path.with_suffix(".jpg")
            image.save(canonical, "JPEG", quality=94, optimize=True)
        if canonical != path:
            path.unlink(missing_ok=True)
            path = canonical
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        path.unlink(missing_ok=True)
        raise MediaValidationError("The upload is not a valid supported image") from exc
    except MediaValidationError:
        path.unlink(missing_ok=True)
        raise
    return StoredMedia(
        path.relative_to(settings.parking_data_root).as_posix(),
        digest,
        size,
        "image/jpeg",
        width,
        height,
    )


async def store_video(upload: UploadFile, settings: Settings) -> StoredMedia:
    suffix = Path(upload.filename or "video").suffix.lower()
    if suffix not in {".mp4", ".avi"}:
        raise MediaValidationError("Supported video containers are MP4 and AVI")
    path, size, digest = await _stream_upload(
        upload, settings.upload_root / "videos", settings.max_video_upload_mb * 1024 * 1024
    )
    with path.open("rb") as handle:
        header = handle.read(32)
    is_mp4 = len(header) >= 12 and header[4:8] == b"ftyp"
    is_avi = len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI "
    if (suffix == ".mp4" and not is_mp4) or (suffix == ".avi" and not is_avi):
        path.unlink(missing_ok=True)
        raise MediaValidationError("Video magic bytes do not match the declared container")
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise MediaValidationError("The video container or codec could not be decoded")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if width <= 0 or height <= 0 or fps <= 0 or frame_count <= 0:
            raise MediaValidationError("Video metadata is incomplete or invalid")
        duration = frame_count / fps
        if width > settings.max_video_width or height > settings.max_video_height:
            raise MediaValidationError("Video resolution exceeds the configured limit")
        if duration > settings.max_video_duration_seconds:
            raise MediaValidationError("Video duration exceeds the configured limit")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise MediaValidationError("The first video frame could not be decoded")
    except MediaValidationError:
        capture.release()
        path.unlink(missing_ok=True)
        raise
    capture.release()
    mime = "video/mp4" if suffix == ".mp4" else "video/x-msvideo"
    return StoredMedia(
        path.relative_to(settings.parking_data_root).as_posix(),
        digest,
        size,
        mime,
        width,
        height,
        round(duration, 3),
        round(fps, 3),
        frame_count,
    )


def resolve_media_path(settings: Settings, relative_path: str) -> Path:
    root = settings.parking_data_root.resolve()
    candidate = (root / relative_path).resolve()
    if root not in candidate.parents:
        raise MediaValidationError("Media path escapes the configured data root")
    return candidate


def cleanup_media_artifacts(settings: Settings, referenced_paths: set[str]) -> dict[str, int]:
    """Delete expired unreferenced artifacts; database-linked results are always preserved."""

    root = settings.media_root.resolve()
    if not root.exists():
        return {"examined": 0, "removed": 0, "preserved": 0}
    cutoff = datetime.now(UTC) - timedelta(days=max(settings.media_retention_days, 1))
    temporary_cutoff = datetime.now(UTC) - timedelta(days=1)
    examined = removed = preserved = 0
    normalized_references = {str(Path(value).as_posix()) for value in referenced_paths}
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        examined += 1
        relative = path.relative_to(settings.parking_data_root).as_posix()
        modified = datetime.fromtimestamp(path.stat().st_mtime, UTC)
        is_temporary = path.suffix in {".upload", ".tmp", ".partial"}
        expired = modified < (temporary_cutoff if is_temporary else cutoff)
        if relative in normalized_references or not expired:
            preserved += 1
            continue
        path.unlink(missing_ok=True)
        removed += 1
    return {"examined": examined, "removed": removed, "preserved": preserved}
