from __future__ import annotations

import re
from fastapi import APIRouter, Query

from app.services.runtime import event_store

router = APIRouter(prefix="/search", tags=["search"])


def parse_query(q: str) -> dict:
    text = q.lower()
    filters: dict = {}
    if "person" in text or "people" in text or "human" in text: filters["object_type"] = "person"
    if "vehicle" in text or "car" in text or "truck" in text: filters["vehicle"] = True
    if "restricted" in text or "intrusion" in text or "fence" in text: filters["event_type"] = "INTRUSION"
    if "high risk" in text or "critical" in text: filters["min_risk"] = 70
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
