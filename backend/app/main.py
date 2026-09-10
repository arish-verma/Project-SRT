from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.alerts import router as alerts_router
from app.api.analytics import router as analytics_router
from app.api.cameras import router as cameras_router
from app.api.events import router as events_router
from app.api.faces import router as faces_router
from app.api.realtime import router as realtime_router
from app.api.search import router as search_router
from app.api.streams import router as streams_router
from app.api.system import router as system_router
from app.api.uploads import router as uploads_router
from app.api.zones import router as zones_router
from app.core.config import settings
from app.core.logging import configure_logging
from app.services.runtime import video_processor

configure_logging(settings.log_level)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    video_processor.stop_all()


app = FastAPI(title=settings.app_name, version=settings.app_version, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in settings.cors_origins.split(",") if x.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
for router in (
    system_router,
    cameras_router,
    streams_router,
    zones_router,
    events_router,
    search_router,
    uploads_router,
    alerts_router,
    analytics_router,
    faces_router,
):
    app.include_router(router, prefix=settings.api_prefix)
app.include_router(realtime_router)


@app.get("/")
def root() -> dict[str, str]:
    # Keep both names for compatibility with early SRT clients/tests.
    return {"name": settings.app_name, "service": settings.app_name, "status": "running", "docs": "/docs"}
