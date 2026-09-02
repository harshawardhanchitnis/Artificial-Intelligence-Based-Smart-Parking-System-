from fastapi import APIRouter

from app.api.routes import analysis, datasets, health, system

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["datasets"])
api_router.include_router(analysis.router, tags=["analysis"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
