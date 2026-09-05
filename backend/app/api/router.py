from fastapi import APIRouter

from app.api.routes import (
    analysis,
    datasets,
    demo,
    diagnostics,
    health,
    insights,
    media,
    system,
    video,
)

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(analysis.router, tags=["analysis"])
api_router.include_router(media.router, prefix="/media", tags=["media"])
api_router.include_router(video.router, prefix="/video", tags=["video"])
api_router.include_router(demo.router, prefix="/demo", tags=["demo"])
api_router.include_router(diagnostics.router, prefix="/diagnostics", tags=["diagnostics"])
api_router.include_router(insights.router, tags=["insights", "reports"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
