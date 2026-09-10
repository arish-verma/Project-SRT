from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.core.logging import configure_logging
from app.api.system import router as system_router

configure_logging(settings.log_level)

app = FastAPI(title=settings.app_name, version=settings.app_version)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[x.strip() for x in settings.cors_origins.split(",") if x.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(system_router, prefix=settings.api_prefix)

@app.get("/")
def root() -> dict[str, str]:
    return {"service": settings.app_name, "status": "running", "docs": "/docs"}
