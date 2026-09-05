from __future__ import annotations

import json

from sqlalchemy import select

from app.core.config import get_settings
from app.db.models import AnalysisRecord, MediaAsset, VideoAnalysis
from app.db.session import SessionLocal
from app.services.media_service import cleanup_media_artifacts


def main() -> int:
    settings = get_settings()
    with SessionLocal() as session:
        referenced = {
            str(value) for value in session.scalars(select(MediaAsset.storage_path)).all() if value
        }
        referenced.update(
            str(value)
            for value in session.scalars(select(AnalysisRecord.result_image_path)).all()
            if value
        )
        referenced.update(
            str(value)
            for value in session.scalars(select(VideoAnalysis.result_video_path)).all()
            if value
        )
    report = cleanup_media_artifacts(settings, referenced)
    print(json.dumps({"retention_days": settings.media_retention_days, **report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
