from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.config import get_settings
from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.migrations import migrate_sqlite

settings = get_settings()
if settings.database_url.startswith("sqlite:///"):
    database_path_text = settings.database_url.removeprefix("sqlite:///")
    Path(database_path_text).parent.mkdir(parents=True, exist_ok=True)

engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def initialize_database() -> dict[str, object]:
    Base.metadata.create_all(bind=engine)
    return migrate_sqlite(engine)
