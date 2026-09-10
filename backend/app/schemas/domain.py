from datetime import datetime, timezone
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, ConfigDict

class Severity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class AlertStatus(str, Enum):
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"

class EventType(str, Enum):
    DETECTION = "DETECTION"
    INTRUSION = "INTRUSION"
    LOITERING = "LOITERING"
    ANPR = "ANPR"
    FACE_MATCH = "FACE_MATCH"
    NIGHT_MOVEMENT = "NIGHT_MOVEMENT"
    SUSPICIOUS_ACTIVITY = "SUSPICIOUS_ACTIVITY"

class ObjectRef(BaseModel):
    type: str
    track_id: int | None = None
    confidence: float | None = Field(default=None, ge=0, le=1)

class ZoneRef(BaseModel):
    id: str
    name: str

class EvidenceRef(BaseModel):
    thumbnail: str | None = None
    video_clip: str | None = None

class SRTEvent(BaseModel):
    model_config = ConfigDict(use_enum_values=True)
    event_id: str
    camera_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: EventType
    severity: Severity = Severity.INFO
    confidence: float = Field(ge=0, le=1)
    risk_score: int = Field(default=0, ge=0, le=100)
    objects: list[ObjectRef] = Field(default_factory=list)
    zone: ZoneRef | None = None
    location: dict[str, Any] = Field(default_factory=dict)
    evidence: EvidenceRef = Field(default_factory=EvidenceRef)
    metadata: dict[str, Any] = Field(default_factory=dict)

class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
