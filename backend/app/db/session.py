from pathlib import Path

from sqlalchemy import create_engine, event
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
    connect_args={"check_same_thread": False, "timeout": 30}
    if settings.database_url.startswith("sqlite")
    else {},
    pool_pre_ping=True,
)


if settings.database_url.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def configure_sqlite(dbapi_connection, _connection_record) -> None:
        cursor = dbapi_connection.cursor()
        try:
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=30000")
            cursor.execute("PRAGMA journal_mode=WAL")
        finally:
            cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


def initialize_database() -> dict[str, object]:
    Base.metadata.create_all(bind=engine)
    return migrate_sqlite(engine)
