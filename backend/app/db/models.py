from datetime import UTC, datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id: Mapped[int] = mapped_column(primary_key=True)
    dataset: Mapped[str] = mapped_column(String(40), index=True)
    scenario_id: Mapped[str] = mapped_column(String(160), index=True)
    total_spaces: Mapped[int] = mapped_column(Integer)
    occupied_spaces: Mapped[int] = mapped_column(Integer)
    vacant_spaces: Mapped[int] = mapped_column(Integer)
    processing_time_ms: Mapped[float] = mapped_column(Float)
    result_image_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(100), nullable=True)
    average_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ground_truth_agreement: Mapped[float | None] = mapped_column(Float, nullable=True)
    prediction_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(String(20), default="scenario")
    media_asset_id: Mapped[int | None] = mapped_column(
        ForeignKey("media_assets.id"), nullable=True, index=True
    )
    job_id: Mapped[int | None] = mapped_column(
        ForeignKey("analysis_jobs.id"), nullable=True, index=True
    )
    layout_id: Mapped[int | None] = mapped_column(
        ForeignKey("detected_layouts.id"), nullable=True, index=True
    )
    localization_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    result_status: Mapped[str] = mapped_column(String(30), default="success")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class MediaAsset(Base):
    __tablename__ = "media_assets"

    id: Mapped[int] = mapped_column(primary_key=True)
    media_type: Mapped[str] = mapped_column(String(20), index=True)
    original_name: Mapped[str] = mapped_column(String(255))
    storage_path: Mapped[str] = mapped_column(String(600), unique=True)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    mime_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int] = mapped_column(Integer)
    width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    fps: Mapped[float | None] = mapped_column(Float, nullable=True)
    frame_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class AnalysisJob(Base):
    __tablename__ = "analysis_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)
    media_asset_id: Mapped[int] = mapped_column(ForeignKey("media_assets.id"), index=True)
    kind: Mapped[str] = mapped_column(String(20), index=True)
    status: Mapped[str] = mapped_column(String(30), index=True, default="queued")
    phase: Mapped[str] = mapped_column(String(80), default="queued")
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, default=False)
    error_code: Mapped[str | None] = mapped_column(String(80), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_analysis_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DetectedLayout(Base):
    __tablename__ = "detected_layouts"

    id: Mapped[int] = mapped_column(primary_key=True)
    media_asset_id: Mapped[int] = mapped_column(ForeignKey("media_assets.id"), index=True)
    model_name: Mapped[str] = mapped_column(String(120))
    layout_identifier: Mapped[str] = mapped_column(String(120), index=True)
    confidence: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(30), default="automatic")
    reference_frame_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)
    geometry_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )


class DetectedSlot(Base):
    __tablename__ = "detected_slots"

    id: Mapped[int] = mapped_column(primary_key=True)
    layout_id: Mapped[int] = mapped_column(ForeignKey("detected_layouts.id"), index=True)
    slot_index: Mapped[int] = mapped_column(Integer)
    polygon_json: Mapped[str] = mapped_column(Text)
    localization_confidence: Mapped[float] = mapped_column(Float)
    corner_confidence: Mapped[float] = mapped_column(Float)


class VideoAnalysis(Base):
    __tablename__ = "video_analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_id: Mapped[int] = mapped_column(ForeignKey("analysis_records.id"), index=True)
    media_asset_id: Mapped[int] = mapped_column(ForeignKey("media_assets.id"), index=True)
    layout_id: Mapped[int | None] = mapped_column(ForeignKey("detected_layouts.id"), nullable=True)
    sample_fps: Mapped[float] = mapped_column(Float)
    processed_frames: Mapped[int] = mapped_column(Integer)
    dropped_frames: Mapped[int] = mapped_column(Integer, default=0)
    stability_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    timeline_json: Mapped[str] = mapped_column(Text)
    result_video_path: Mapped[str | None] = mapped_column(String(600), nullable=True)


class OccupancyEvent(Base):
    __tablename__ = "occupancy_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    video_analysis_id: Mapped[int] = mapped_column(ForeignKey("video_analyses.id"), index=True)
    slot_index: Mapped[int] = mapped_column(Integer)
    timestamp_seconds: Mapped[float] = mapped_column(Float)
    previous_state: Mapped[str] = mapped_column(String(20))
    next_state: Mapped[str] = mapped_column(String(20))
    confidence: Mapped[float] = mapped_column(Float)


class LayoutCorrection(Base):
    __tablename__ = "layout_corrections"

    id: Mapped[int] = mapped_column(primary_key=True)
    layout_id: Mapped[int] = mapped_column(ForeignKey("detected_layouts.id"), index=True)
    before_json: Mapped[str] = mapped_column(Text)
    after_json: Mapped[str] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
