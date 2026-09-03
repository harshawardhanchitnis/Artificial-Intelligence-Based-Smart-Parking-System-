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


@asynccontextmanager
async def lifespan(_: FastAPI):
    initialize_database()
    yield


settings = get_settings()
app = FastAPI(
    title=settings.app_name,
    version="0.7.0",
    description="Offline dataset-driven smart-parking API",
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
    return {"name": settings.app_name, "status": "running", "docs": "/docs"}
