from __future__ import annotations

from datetime import datetime
from enum import Enum
from pydantic import BaseModel

from app.schemas.event import EventSeverity


class AlertStatus(str, Enum):
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class AlertRecord(BaseModel):
    alert_id: str
    event_id: str
    camera_id: str
    created_at: datetime
    severity: EventSeverity
    title: str
    message: str
    status: AlertStatus = AlertStatus.NEW


class AlertUpdate(BaseModel):
    status: AlertStatus
