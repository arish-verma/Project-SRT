from fastapi import APIRouter

from app.services.runtime import camera_manager, event_store

router = APIRouter(prefix="/analytics", tags=["analytics"])

@router.get("/summary")
def summary():
    cameras = camera_manager.list()
    online = sum(1 for c in cameras if c.status.value == "ONLINE")
    events = event_store.list(limit=500)
    return {
        "cameras_total": len(cameras),
        "cameras_online": online,
        "events_total": event_store.count(),
        "events_recent": len(events),
        "high_priority": sum(1 for e in events if e.severity.value in {"HIGH", "CRITICAL"}),
        "intrusions": sum(1 for e in events if e.event_type == "INTRUSION"),
        "by_type": {k: sum(1 for e in events if e.event_type == k) for k in sorted({e.event_type for e in events})},
    }
