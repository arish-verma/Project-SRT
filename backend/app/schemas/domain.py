from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class Severity(str, Enum):
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class AlertState(str, Enum):
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class Camera(BaseModel):
    camera_id: str
    name: str
    source: str
    location: str | None = None
    enabled: bool = True


class Detection(BaseModel):
    object_type: str
    confidence: float = Field(ge=0, le=1)
    bbox: tuple[float, float, float, float]


class Track(BaseModel):
    track_id: int
    object_type: str
    confidence: float = Field(ge=0, le=1)
    bbox: tuple[float, float, float, float]


class Zone(BaseModel):
    zone_id: str
    name: str
    polygon: list[tuple[float, float]]
    restricted: bool = True
    severity: Severity = Severity.HIGH


class Event(BaseModel):
    event_id: str
    camera_id: str
    timestamp: datetime
    event_type: str
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    risk_score: int = Field(ge=0, le=100)
    objects: list[dict] = []
    zone: dict | None = None
    location: dict | None = None
    evidence: dict | None = None
    metadata: dict = {}


class Alert(BaseModel):
    alert_id: str
    event_id: str
    state: AlertState = AlertState.NEW
    severity: Severity
    created_at: datetime
