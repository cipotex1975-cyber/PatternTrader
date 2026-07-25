from fastapi import APIRouter

from app.core.config.settings import get_settings

router = APIRouter()


@router.get("/health")
async def health_check():
    settings = get_settings()
    return {
        "status": "healthy",
        "application": settings.application.name,
        "version": settings.application.version,
        "environment": settings.application.environment,
    }


@router.get("/info")
async def get_info():
    settings = get_settings()
    return {
        "name": settings.application.name,
        "version": settings.application.version,
        "debug": settings.application.debug,
    }
