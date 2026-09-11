from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.core.config import settings
from app.services.runtime import alert_store, event_store

router = APIRouter(prefix="/events", tags=["events"])


@router.get("", response_model=list)
def list_events(camera_id: str | None = None, event_type: str | None = None, limit: int = 100):
    return event_store.list(camera_id=camera_id, event_type=event_type, limit=limit)


@router.get("/evidence/{event_id}")
def evidence(event_id: str):
    event = next((x for x in event_store.list(limit=500) if x.event_id == event_id), None)
    if not event or not event.evidence_frame:
        raise HTTPException(404, "Evidence not found")
    path = Path(event.evidence_frame)
    if not path.exists():
        raise HTTPException(404, "Evidence unavailable")
    return FileResponse(path, media_type="image/jpeg", filename=f"{event_id}.jpg")


@router.get("/recording/{event_id}")
def recording(event_id: str):
    event = next((x for x in event_store.list(limit=500) if x.event_id == event_id), None)
    path = Path((event.metadata or {}).get("recording_clip", "")) if event else Path("")
    if not event or not path.exists():
        raise HTTPException(404, "Event recording unavailable")
    return FileResponse(path, media_type="video/mp4", filename=f"{event_id}.mp4")


@router.delete("/all", status_code=204)
def clear_events():
    event_store.clear()
    # Events and alerts represent one incident stream. Clearing events must
    # not leave orphaned alerts that can appear to belong to a new event.
    alert_store.clear()


@router.delete("/{event_id}", status_code=204)
def delete_event(event_id: str):
    if not event_store.delete(event_id):
        raise HTTPException(404, "Event not found")
    alert_store.delete_for_event(event_id)
