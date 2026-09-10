from __future__ import annotations

from enum import Enum
from pydantic import BaseModel, Field


class ZoneType(str, Enum):
    RESTRICTED = "RESTRICTED"
    MONITORED = "MONITORED"
    EXCLUSION = "EXCLUSION"


class ZoneCreate(BaseModel):
    camera_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=120)
    zone_type: ZoneType = ZoneType.RESTRICTED
    polygon: list[tuple[float, float]] = Field(min_length=3)
    severity: str = "HIGH"
    enabled: bool = True


class Zone(ZoneCreate):
    zone_id: str
