from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.cameras import router as cameras_router
from app.api.events import router as events_router
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
app.include_router(system_router, prefix=settings.api_prefix)
app.include_router(cameras_router, prefix=settings.api_prefix)
app.include_router(streams_router, prefix=settings.api_prefix)
app.include_router(zones_router, prefix=settings.api_prefix)
app.include_router(events_router, prefix=settings.api_prefix)
app.include_router(search_router, prefix=settings.api_prefix)
app.include_router(uploads_router, prefix=settings.api_prefix)
app.include_router(realtime_router)


@app.get("/")
def root() -> dict[str, str]:
    return {"service": settings.app_name, "status": "running", "docs": "/docs"}
