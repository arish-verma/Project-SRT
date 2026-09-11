from fastapi import APIRouter, HTTPException

from app.schemas.event import EventRecord
from app.services.runtime import event_store

router = APIRouter(prefix="/events", tags=["events"])

@router.get("", response_model=list[EventRecord])
def list_events(camera_id: str | None = None, event_type: str | None = None, limit: int = 100) -> list[EventRecord]:
    return event_store.list(camera_id=camera_id, event_type=event_type, limit=limit)

@router.delete("/{event_id}", status_code=204)
def delete_event(event_id: str) -> None:
    if not event_store.delete(event_id):
        raise HTTPException(status_code=404, detail="Event not found")
