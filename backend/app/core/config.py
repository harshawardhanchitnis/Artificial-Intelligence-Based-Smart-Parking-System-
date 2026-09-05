from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATA_ROOT = Path("D:/Projects/AI Based Smart Parking System Data")
DEFAULT_DATABASE_PATH = PROJECT_ROOT / "data" / "app" / "smart_parking.db"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Artificial Intelligence Based Smart Parking System"
    app_env: str = "development"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    parking_data_root: Path = DEFAULT_DATA_ROOT
    database_url: str = f"sqlite:///{DEFAULT_DATABASE_PATH.as_posix()}"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    max_image_upload_mb: int = 25
    max_video_upload_mb: int = 500
    max_image_dimension: int = 8192
    max_video_duration_seconds: int = 300
    max_video_width: int = 1920
    max_video_height: int = 1080
    video_sample_fps: float = 2.0
    media_retention_days: int = 30

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def archives_root(self) -> Path:
        return self.parking_data_root / "archives"

    @property
    def model_root(self) -> Path:
        return self.parking_data_root / "models"

    @property
    def media_root(self) -> Path:
        return self.parking_data_root / "media"

    @property
    def upload_root(self) -> Path:
        return self.media_root / "uploads"

    @property
    def result_root(self) -> Path:
        return self.media_root / "results"

    @property
    def job_root(self) -> Path:
        return self.media_root / "jobs"


@lru_cache
def get_settings() -> Settings:
    return Settings()
