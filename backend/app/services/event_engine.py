from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any

import cv2

from app.ai.interfaces import Detection, Track
from app.schemas.event import EventRecord, EventSeverity
from app.schemas.zone import Zone


class RuleEventEngine:
    """Explainable spatial/temporal event engine for the hackathon prototype."""

    def __init__(self, cooldown_seconds: float = 8.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._inside: set[tuple[str, int, str]] = set()
        self._last_event: dict[tuple[Any, ...], float] = {}

    @staticmethod
    def _contains_point(zone: Zone, x: float, y: float, width: int, height: int) -> bool:
        point = (float(x), float(y))
        polygon = [(px * width, py * height) for px, py in zone.polygon]
        return cv2.pointPolygonTest(__import__("numpy").array(polygon, dtype="float32"), point, False) >= 0

    @classmethod
    def _contains(cls, zone: Zone, track: Track, width: int, height: int) -> bool:
        x, y = track.center
        return cls._contains_point(zone, x, y, width, height)

    def evaluate(self, camera_id: str, tracks: list[Track], zones: list[Zone], frame_shape: tuple[int, ...], timestamp: float | None = None) -> list[EventRecord]:
        now = timestamp or time.time()
        height, width = frame_shape[:2]
        events: list[EventRecord] = []
        current: set[tuple[str, int, str]] = set()
        for zone in zones:
            if not zone.enabled or zone.camera_id != camera_id:
                continue
            for track in tracks:
                if track.label not in {"person", "car", "truck", "bus", "motorcycle"}:
                    continue
                key = (camera_id, track.track_id, zone.zone_id)
                inside = self._contains(zone, track, width, height)
                if inside:
                    current.add(key)
                if not inside or key in self._inside:
                    continue
                last = self._last_event.get(key, 0.0)
                if now - last < self.cooldown_seconds:
                    continue
                self._last_event[key] = now
                risk = 82 if zone.zone_type.value == "RESTRICTED" and track.label == "person" else 65
                severity = EventSeverity.HIGH if risk >= 75 else EventSeverity.MEDIUM
                events.append(EventRecord(
                    event_id=f"EVT-{uuid.uuid4().hex[:10].upper()}", camera_id=camera_id,
                    timestamp=datetime.fromtimestamp(now, tz=timezone.utc), event_type="INTRUSION",
                    severity=severity, confidence=track.confidence, risk_score=risk,
                    object_type=track.label, track_id=track.track_id, zone_id=zone.zone_id,
                    zone_name=zone.name, message=f"{track.label.title()} entered {zone.name}",
                    metadata={"zone_type": zone.zone_type.value},
                ))
        self._inside = current
        return events

    def evaluate_drones(self, camera_id: str, detections: list[Detection], zones: list[Zone], frame_shape: tuple[int, ...], timestamp: float | None = None) -> list[EventRecord]:
        """Create throttled drone events; restricted-zone presence raises severity."""
        now = timestamp or time.time()
        height, width = frame_shape[:2]
        events: list[EventRecord] = []
        restricted_zone: Zone | None = None
        for zone in zones:
            if zone.enabled and zone.camera_id == camera_id and zone.zone_type.value == "RESTRICTED":
                restricted_zone = zone
                break

        for detection in detections:
            x1, y1, x2, y2 = detection.bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            in_restricted = bool(restricted_zone and self._contains_point(restricted_zone, cx, cy, width, height))
            bucket = (camera_id, "drone", round(cx / max(width, 1), 1), round(cy / max(height, 1), 1))
            last = self._last_event.get(bucket, 0.0)
            if now - last < max(self.cooldown_seconds, 10.0):
                continue
            self._last_event[bucket] = now
            risk = 90 if in_restricted else 60
            severity = EventSeverity.HIGH if in_restricted else EventSeverity.MEDIUM
            zone_name = restricted_zone.name if in_restricted and restricted_zone else None
            message = "Drone detected in restricted zone" if in_restricted else "Drone detected"
            events.append(EventRecord(
                event_id=f"EVT-{uuid.uuid4().hex[:10].upper()}", camera_id=camera_id,
                timestamp=datetime.fromtimestamp(now, tz=timezone.utc), event_type="DRONE_DETECTED",
                severity=severity, confidence=detection.confidence, risk_score=risk,
                object_type="drone", zone_id=restricted_zone.zone_id if in_restricted and restricted_zone else None,
                zone_name=zone_name, message=message,
                metadata={"restricted_zone": in_restricted, "bbox": list(detection.bbox)},
            ))
        return events
