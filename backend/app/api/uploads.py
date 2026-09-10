from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings

router = APIRouter(prefix="/uploads", tags=["uploads"])
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


@router.post("/video")
async def upload_video(file: UploadFile = File(...)) -> dict[str, str]:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported video format")
    root = Path(settings.upload_storage_path)
    root.mkdir(parents=True, exist_ok=True)
    target = root / f"{uuid.uuid4().hex}{ext}"
    size = 0
    try:
        with target.open("wb") as handle:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > settings.max_upload_size_mb * 1024 * 1024:
                    raise HTTPException(status_code=413, detail="Video exceeds configured upload limit")
                handle.write(chunk)
    except HTTPException:
        target.unlink(missing_ok=True)
        raise
    except Exception:
        target.unlink(missing_ok=True)
        raise
    finally:
        await file.close()
    return {"source": str(target), "filename": file.filename or target.name, "size_bytes": str(size)}
