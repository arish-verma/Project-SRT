from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.system import router as system_router
from app.api.routes.cameras import router as cameras_router
from app.core.config import settings

app = FastAPI(
    title="Project SRT API",
    version="0.1.0",
    description="Smart Recognition & Tracking video analytics platform API",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(system_router, prefix="/api/v1/system", tags=["system"])
app.include_router(cameras_router, prefix="/api/v1/cameras", tags=["cameras"])


@app.get("/", tags=["system"])
def root() -> dict[str, str]:
    return {"name": "Project SRT", "version": app.version, "status": "online"}
