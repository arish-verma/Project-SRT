from fastapi import APIRouter
from app.schemas.domain import HealthResponse
from app.core.config import settings

router = APIRouter(prefix="/system", tags=["system"])

@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", service=settings.app_name, version=settings.app_version)
