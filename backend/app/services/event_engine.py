from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import cv2

from app.ai.interfaces import Detection, Track
from app.schemas.event import EventRecord, EventSeverity
from app.schemas.zone import Zone


class RuleEventEngine:
    """Explainable spatial + temporal event reasoning for the surveillance prototype."""

    def __init__(self, cooldown_seconds: float = 8.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._inside: set[tuple[str, int, str]] = set()
        self._last_event: dict[tuple[Any, ...], float] = {}
        self._entered_at: dict[tuple[str, int, str], float] = {}
        self._history: dict[tuple[str, int], deque[tuple[float, float, float]]] = defaultdict(lambda: deque(maxlen=24))
        self._drone_tracks: dict[str, dict[int, tuple[float, float, float, float]]] = defaultdict(dict)
        self._next_drone_id: dict[str, int] = defaultdict(lambda: 1)

    @staticmethod
    def _contains_point(zone: Zone, x: float, y: float, width: int, height: int) -> bool:
        point = (float(x), float(y))
        polygon = [(px * width, py * height) for px, py in zone.polygon]
        return cv2.pointPolygonTest(__import__("numpy").array(polygon, dtype="float32"), point, False) >= 0

    @classmethod
    def _contains(cls, zone: Zone, track: Track, width: int, height: int) -> bool:
        x, y = track.center
        return cls._contains_point(zone, x, y, width, height)

    @staticmethod
    def _is_night(timestamp: float) -> bool:
        # Uses the host's local camera/server timezone so deployments can configure
        # timezone at the OS/container level instead of hard-coding a country.
        hour = datetime.fromtimestamp(timestamp).hour
        return hour >= 22 or hour < 5

    @staticmethod
    def _direction(history: deque[tuple[float, float, float]]) -> str | None:
        if len(history) < 3:
            return None
        _, x0, y0 = history[0]
        _, x1, y1 = history[-1]
        dx, dy = x1 - x0, y1 - y0
        if math.hypot(dx, dy) < 8:
            return "STATIONARY"
        horizontal = "EAST" if dx > 0 else "WEST"
        vertical = "SOUTH" if dy > 0 else "NORTH"
        if abs(dx) < abs(dy) * 0.5:
            return vertical
        if abs(dy) < abs(dx) * 0.5:
            return horizontal
        return f"{vertical}-{horizontal}"

    @staticmethod
    def _movement_distance(history: deque[tuple[float, float, float]]) -> float:
        if len(history) < 2:
            return 0.0
        return sum(math.hypot(history[i][1] - history[i - 1][1], history[i][2] - history[i - 1][2])
                   for i in range(1, len(history)))

    def _new_event(self, camera_id: str, now: float, event_type: str, severity: EventSeverity,
                   confidence: float, risk_score: int, message: str, object_type: str = "person",
                   track_id: int | None = None, zone: Zone | None = None, metadata: dict | None = None) -> EventRecord:
        return EventRecord(
            event_id=f"EVT-{uuid.uuid4().hex[:10].upper()}", camera_id=camera_id,
            timestamp=datetime.fromtimestamp(now, tz=timezone.utc), event_type=event_type,
            severity=severity, confidence=confidence, risk_score=min(max(risk_score, 0), 100),
            object_type=object_type, track_id=track_id,
            zone_id=zone.zone_id if zone else None, zone_name=zone.name if zone else None,
            message=message, metadata=metadata or {},
        )

    def _allow(self, key: tuple[Any, ...], now: float, cooldown: float | None = None) -> bool:
        last = self._last_event.get(key, 0.0)
        if now - last < (cooldown if cooldown is not None else self.cooldown_seconds):
            return False
        self._last_event[key] = now
        return True

    def evaluate(self, camera_id: str, tracks: list[Track], zones: list[Zone], frame_shape: tuple[int, ...], timestamp: float | None = None) -> list[EventRecord]:
        now = timestamp or time.time()
        height, width = frame_shape[:2]
        events: list[EventRecord] = []
        current: set[tuple[str, int, str]] = set()
        night = self._is_night(now)

        for track in tracks:
            self._history[(camera_id, track.track_id)].append((now, track.center[0], track.center[1]))

        for zone in zones:
            if not zone.enabled or zone.camera_id != camera_id:
                continue
            persons_inside = 0
            for track in tracks:
                if track.label not in {"person", "car", "truck", "bus", "motorcycle"}:
                    continue
                key = (camera_id, track.track_id, zone.zone_id)
                inside = self._contains(zone, track, width, height)
                if inside:
                    current.add(key)
                    persons_inside += track.label == "person"
                    self._entered_at.setdefault(key, now)

                if not inside:
                    self._entered_at.pop(key, None)
                    continue

                history = self._history[(camera_id, track.track_id)]
                direction = self._direction(history)
                dwell = max(0.0, now - self._entered_at.get(key, now))
                entry = key not in self._inside

                if entry and self._allow(key, now):
                    # Keep the proven hackathon UX: restricted person entry is a HIGH event.
                    risk = 82 if zone.zone_type.value == "RESTRICTED" and track.label == "person" else 65
                    factors = ["restricted zone entry"] if zone.zone_type.value == "RESTRICTED" else ["monitored zone entry"]
                    if night and track.label == "person":
                        risk = min(100, risk + 10)
                        factors.append("night-time")
                    if direction and direction != "STATIONARY":
                        factors.append(f"direction {direction.lower()}")
                    severity = EventSeverity.CRITICAL if risk >= 90 else (EventSeverity.HIGH if risk >= 75 else EventSeverity.MEDIUM)
                    events.append(self._new_event(
                        camera_id, now, "INTRUSION", severity, track.confidence, risk,
                        f"{track.label.title()} entered {zone.name}", track.label, track.track_id, zone,
                        {"zone_type": zone.zone_type.value, "factors": factors, "night": night,
                         "dwell_seconds": round(dwell, 1), "direction": direction},
                    ))

                if track.label == "person" and zone.zone_type.value == "RESTRICTED":
                    if dwell >= 30 and self._allow((camera_id, track.track_id, zone.zone_id, "loiter"), now, 15):
                        movement = self._movement_distance(history)
                        if movement < max(width, height) * 0.18:
                            events.append(self._new_event(
                                camera_id, now, "LOITERING", EventSeverity.HIGH, track.confidence, 80,
                                f"Person remained in {zone.name} for {int(dwell)}s with limited movement",
                                "person", track.track_id, zone,
                                {"dwell_seconds": round(dwell, 1), "movement_pixels": round(movement, 1),
                                 "factors": ["restricted zone", "extended dwell", "limited movement"]},
                            ))
                    if night and self._allow((camera_id, track.track_id, zone.zone_id, "night"), now, 20):
                        events.append(self._new_event(
                            camera_id, now, "NIGHT_MOVEMENT", EventSeverity.HIGH, track.confidence, 75,
                            f"Night-time movement detected in {zone.name}", "person", track.track_id, zone,
                            {"factors": ["night-time", "restricted zone"], "direction": direction},
                        ))

            if persons_inside >= 3 and self._allow((camera_id, zone.zone_id, "group"), now, 20):
                events.append(self._new_event(
                    camera_id, now, "MULTI_PERSON_ACTIVITY", EventSeverity.HIGH, 0.9, 78,
                    f"Multiple people detected in {zone.name}", "person", None, zone,
                    {"person_count": persons_inside, "factors": ["restricted zone", "multiple people"]},
                ))

        self._inside = current
        # Prevent stale history from growing forever when a track disappears.
        active_ids = {(camera_id, t.track_id) for t in tracks}
        for key in list(self._history):
            if key[0] == camera_id and key not in active_ids and self._history[key]:
                if now - self._history[key][-1][0] > 10:
                    del self._history[key]
        return events

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / max(area_a + area_b - inter, 1e-6)

    def _assign_drone_tracks(self, camera_id: str, detections: list[Detection]) -> list[tuple[int, Detection]]:
        previous = self._drone_tracks[camera_id]
        result: list[tuple[int, Detection]] = []
        used: set[int] = set()
        for detection in detections:
            best_id, best_iou = None, 0.20
            for track_id, bbox in previous.items():
                if track_id in used:
                    continue
                score = self._iou(bbox, detection.bbox)
                if score > best_iou:
                    best_id, best_iou = track_id, score
            if best_id is None:
                best_id = self._next_drone_id[camera_id]
                self._next_drone_id[camera_id] += 1
            used.add(best_id)
            result.append((best_id, detection))
        self._drone_tracks[camera_id] = {track_id: detection.bbox for track_id, detection in result}
        return result

    def evaluate_drones(self, camera_id: str, detections: list[Detection], zones: list[Zone], frame_shape: tuple[int, ...], timestamp: float | None = None) -> list[EventRecord]:
        """Track drones across specialist detections and create throttled events."""
        now = timestamp or time.time()
        height, width = frame_shape[:2]
        events: list[EventRecord] = []
        restricted_zone: Zone | None = next((z for z in zones if z.enabled and z.camera_id == camera_id and z.zone_type.value == "RESTRICTED"), None)
        for drone_track_id, detection in self._assign_drone_tracks(camera_id, detections):
            x1, y1, x2, y2 = detection.bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            in_restricted = bool(restricted_zone and self._contains_point(restricted_zone, cx, cy, width, height))
            bucket = (camera_id, "drone", drone_track_id, "presence")
            if not self._allow(bucket, now, max(self.cooldown_seconds, 10.0)):
                continue
            risk = 90 if in_restricted else 60
            severity = EventSeverity.CRITICAL if in_restricted else EventSeverity.MEDIUM
            zone_name = restricted_zone.name if in_restricted and restricted_zone else None
            message = "Drone detected in restricted zone" if in_restricted else "Drone detected"
            events.append(self._new_event(
                camera_id, now, "DRONE_DETECTED", severity, detection.confidence, risk, message,
                "drone", None, restricted_zone if in_restricted else None,
                {"restricted_zone": in_restricted, "drone_track_id": drone_track_id,
                 "bbox": list(detection.bbox), "factors": ["drone detection"] + (["restricted zone"] if in_restricted else [])},
            ))
        return events
