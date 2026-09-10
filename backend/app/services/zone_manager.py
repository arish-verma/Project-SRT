from __future__ import annotations

import uuid

from app.schemas.zone import Zone, ZoneCreate


class ZoneManager:
    def __init__(self) -> None:
        self._zones: dict[str, Zone] = {}

    def list(self, camera_id: str | None = None) -> list[Zone]:
        values = list(self._zones.values())
        return [z for z in values if camera_id is None or z.camera_id == camera_id]

    def get(self, zone_id: str) -> Zone | None:
        return self._zones.get(zone_id)

    def create(self, data: ZoneCreate) -> Zone:
        zone = Zone(zone_id=f"ZONE-{uuid.uuid4().hex[:8].upper()}", **data.model_dump())
        self._zones[zone.zone_id] = zone
        return zone

    def delete(self, zone_id: str) -> bool:
        return self._zones.pop(zone_id, None) is not None
