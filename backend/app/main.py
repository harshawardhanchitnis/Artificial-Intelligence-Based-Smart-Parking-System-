import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.router import api_router
from app.core.config import get_settings
from app.core.http import (
    RequestContextMiddleware,
    http_exception_handler,
    validation_exception_handler,
)
from app.db.session import initialize_database
from app.ml.localization import SlotLocalizerPredictor
from app.ml.occupancy_v3 import OccupancyV3Predictor
from app.release import APPLICATION_VERSION
from app.services.video_service import recover_interrupted_jobs


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    recover_interrupted_jobs()
    warm_models()
    yield


def warm_models() -> None:
    """Build the inference sessions once at startup.

    Without this the first analysis of a session pays the model-construction
    cost and is by far the slowest, which reads as the product being slow.
    A missing model is not fatal here -- readiness reports it instead.
    """
    settings = get_settings()
    for load in (OccupancyV3Predictor.load, SlotLocalizerPredictor.load):
        try:
            load(settings.model_root)
        except Exception:  # noqa: BLE001 - readiness surfaces the detail
            logging.getLogger(__name__).info(
                "Model warm-up skipped for %s; readiness will report it", load.__qualname__
            )


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version=APPLICATION_VERSION,
    description="Local image and video parking-occupancy analysis API",
    lifespan=lifespan,
)
app.add_exception_handler(StarletteHTTPException, http_exception_handler)
app.add_exception_handler(RequestValidationError, validation_exception_handler)
app.add_middleware(RequestContextMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(api_router, prefix="/api/v1")


@app.get("/", tags=["root"])
def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "version": APPLICATION_VERSION,
        "status": "running",
        "docs": "/docs",
    }
