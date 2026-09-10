from fastapi import APIRouter, Query

from app.services.runtime import event_store

router = APIRouter(prefix="/search", tags=["search"])


def parse_query(q: str) -> dict:
    text = q.lower().strip()
    filters: dict = {}
    if "person" in text or "people" in text or "human" in text:
        filters["object_type"] = "person"
    elif "drone" in text:
        filters["object_type"] = "drone"
    elif any(word in text for word in ("vehicle", "car", "truck", "bus", "motorcycle")):
        filters["vehicle"] = True
    if "restricted" in text or "intrusion" in text or "fence" in text:
        filters["event_type"] = "INTRUSION"
    elif "loiter" in text or "loitering" in text:
        filters["event_type"] = "LOITERING"
    elif "night" in text or "night-time" in text:
        filters["event_type"] = "NIGHT_MOVEMENT"
    elif "drone" in text:
        filters["event_type"] = "DRONE_DETECTED"
    elif "multiple people" in text or "group" in text:
        filters["event_type"] = "MULTI_PERSON_ACTIVITY"
    if "high risk" in text or "critical" in text or "danger" in text:
        filters["min_risk"] = 70
    if "medium risk" in text:
        filters["min_risk"] = 45
    if "low risk" in text:
        filters["min_risk"] = 20
    return filters


@router.get("")
def natural_language_search(q: str = Query(min_length=1, max_length=300), limit: int = 100):
    filters = parse_query(q)
    events = event_store.list(event_type=filters.get("event_type"), limit=min(limit, 500))
    if filters.get("object_type"):
        events = [e for e in events if e.object_type == filters["object_type"]]
    if filters.get("vehicle"):
        events = [e for e in events if e.object_type in {"car", "truck", "bus", "motorcycle"}]
    if filters.get("min_risk"):
        events = [e for e in events if e.risk_score >= filters["min_risk"]]
    return {"query": q, "interpreted_filters": filters, "results": events}
