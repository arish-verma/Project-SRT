from fastapi import APIRouter

from app.schemas.event import EventRecord
from app.services.event_store import EventStore

router = APIRouter(prefix="/events", tags=["events"])
event_store = EventStore()


@router.get("", response_model=list[EventRecord])
def list_events(camera_id: str | None = None, event_type: str | None = None, limit: int = 100) -> list[EventRecord]:
    return event_store.list(camera_id=camera_id, event_type=event_type, limit=limit)
