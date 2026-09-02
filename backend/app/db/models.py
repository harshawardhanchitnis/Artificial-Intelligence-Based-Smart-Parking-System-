from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, String
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
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(UTC)
    )
