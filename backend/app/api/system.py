from fastapi import APIRouter
from app.schemas.domain import HealthResponse
from app.core.config import settings

router = APIRouter(prefix="/system", tags=["system"])


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # "healthy" is the public contract used by monitoring and the original API tests.
    return HealthResponse(status="healthy", service=settings.app_name, version=settings.app_version)
