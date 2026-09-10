from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.cameras import router as cameras_router, video_processor
from app.api.system import router as system_router
from app.core.config import settings
from app.core.logging import configure_logging

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

@app.get("/")
def root() -> dict[str, str]:
    return {"service": settings.app_name, "status": "running", "docs": "/docs"}
