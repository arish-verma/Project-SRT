import re

from fastapi import APIRouter, Query

from app.services.runtime import camera_manager, event_store

router = APIRouter(prefix="/search", tags=["search"])


def parse_query(q):
    text = q.lower().strip()
    filters = {}

    if any(word in text for word in ("person", "people", "human")):
        filters["object_type"] = "person"
    elif "drone" in text:
        filters["object_type"] = "drone"
        filters["event_type"] = "DRONE_DETECTED"
    elif "helicopter" in text:
        filters["object_type"] = "helicopter"
    elif any(word in text for word in ("airplane", "aircraft", "plane")):
        filters["object_type"] = "airplane"
    elif any(word in text for word in ("aerial object", "aerial")):
        filters["object_type"] = "aerial_object"
    elif any(word in text for word in ("vehicle", "car", "truck", "bus", "motorcycle")):
        filters["vehicle"] = True

    if any(word in text for word in ("restricted", "intrusion", "fence")):
        filters["event_type"] = "INTRUSION"
    elif any(word in text for word in ("loiter", "dwell")):
        filters["event_type"] = "LOITERING"
    elif any(word in text for word in ("night", "night-time", "nighttime")):
        filters["event_type"] = "NIGHT_MOVEMENT"
    elif any(word in text for word in ("fight", "fighting", "altercation", "physical")):
        filters["event_type"] = "FIGHT_SUSPECTED"
    elif "drone" in text:
        filters["event_type"] = "DRONE_DETECTED"
    elif any(word in text for word in ("multiple people", "group", "crowd")):
        filters["event_type"] = "MULTI_PERSON_ACTIVITY"
    elif any(word in text for word in ("anomaly", "unusual", "suspicious")):
        filters["event_type"] = "ANOMALY_SUSPECTED"

    match = re.search(r"camera\s*(\d+)", text)
    if match:
        number = int(match.group(1))
        cameras = camera_manager.list()
        if 1 <= number <= len(cameras):
            filters["camera_id"] = cameras[number - 1].camera_id

    if any(word in text for word in ("high risk", "critical", "danger", "serious")):
        filters["min_risk"] = 70
    elif "medium risk" in text:
        filters["min_risk"] = 45
    elif "low risk" in text:
        filters["min_risk"] = 20
    return filters


@router.get("")
def natural_language_search(q: str = Query(min_length=1, max_length=300), limit: int = 100):
    filters = parse_query(q)
    events = event_store.list(
        camera_id=filters.get("camera_id"),
        event_type=filters.get("event_type"),
        limit=min(limit, 500),
    )
    if filters.get("object_type"):
        events = [event for event in events if event.object_type == filters["object_type"]]
    if filters.get("vehicle"):
        events = [event for event in events if event.object_type in {"car", "truck", "bus", "motorcycle"}]
    if filters.get("min_risk"):
        events = [event for event in events if event.risk_score >= filters["min_risk"]]
    return {
        "query": q,
        "interpreted_filters": filters,
        "results": [
            event.model_dump(mode="json")
            | {"evidence_url": f"/api/v1/events/evidence/{event.event_id}" if event.evidence_frame else None}
            for event in events
        ],
    }
