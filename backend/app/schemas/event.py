from __future__ import annotations

from datetime import datetime
from enum import Enum
from pydantic import BaseModel


class EventSeverity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class EventRecord(BaseModel):
    event_id: str
    camera_id: str
    timestamp: datetime
    event_type: str
    severity: EventSeverity
    confidence: float = 0.0
    risk_score: int = 0
    object_type: str = ""
    track_id: int | None = None
    zone_id: str | None = None
    zone_name: str | None = None
    message: str
    evidence_frame: str | None = None
    metadata: dict = {}
