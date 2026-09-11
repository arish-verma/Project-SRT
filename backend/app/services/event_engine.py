from __future__ import annotations

import math
import time
import uuid
from collections import defaultdict, deque
from datetime import datetime, timezone

import cv2

from app.ai.interfaces import Track
from app.schemas.event import EventRecord, EventSeverity


class RuleEventEngine:
    def __init__(self, cooldown_seconds: float = 8.0) -> None:
        self.cooldown_seconds = cooldown_seconds
        self._inside = set()
        self._last_event = {}
        self._entered_at = {}
        self._history = defaultdict(lambda: deque(maxlen=24))
        self._drone_tracks = defaultdict(dict)
        self._next_drone_id = defaultdict(lambda: 1)
        self._drone_confirmation = defaultdict(dict)
        self._last_drone_scan = {}

    @staticmethod
    def _contains_point(zone, x, y, width, height):
        return cv2.pointPolygonTest(
            __import__("numpy").array([(px * width, py * height) for px, py in zone.polygon], dtype="float32"),
            (float(x), float(y)),
            False,
        ) >= 0

    @staticmethod
    def _is_night(timestamp):
        h = datetime.fromtimestamp(timestamp).hour
        return h >= 22 or h < 5

    @staticmethod
    def _direction(history):
        if len(history) < 3:
            return None
        _, x0, y0 = history[0]
        _, x1, y1 = history[-1]
        dx, dy = x1 - x0, y1 - y0
        if math.hypot(dx, dy) < 8:
            return "STATIONARY"
        h = "EAST" if dx > 0 else "WEST"
        v = "SOUTH" if dy > 0 else "NORTH"
        return v if abs(dx) < abs(dy) * 0.5 else h if abs(dy) < abs(dx) * 0.5 else f"{v}-{h}"

    @staticmethod
    def _movement_distance(history):
        return 0.0 if len(history) < 2 else sum(
            math.hypot(history[i][1] - history[i - 1][1], history[i][2] - history[i - 1][2])
            for i in range(1, len(history))
        )

    def _new_event(self, camera_id, now, event_type, severity, confidence, risk, message,
                   object_type="person", track_id=None, zone=None, metadata=None):
        return EventRecord(
            event_id=f"EVT-{uuid.uuid4().hex[:10].upper()}",
            camera_id=camera_id,
            timestamp=datetime.fromtimestamp(now, tz=timezone.utc),
            event_type=event_type,
            severity=severity,
            confidence=confidence,
            risk_score=min(max(risk, 0), 100),
            object_type=object_type,
            track_id=track_id,
            zone_id=zone.zone_id if zone else None,
            zone_name=zone.name if zone else None,
            message=message,
            metadata=metadata or {},
        )

    def _allow(self, key, now, cooldown=None):
        if now - self._last_event.get(key, 0) < (cooldown if cooldown is not None else self.cooldown_seconds):
            return False
        self._last_event[key] = now
        return True

    def _fight_events(self, camera_id, persons, now, width, height):
        events = []
        for i, a in enumerate(persons):
            for b in persons[i + 1 :]:
                dist = math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])
                ah = max(10.0, a.bbox[3] - a.bbox[1])
                bh = max(10.0, b.bbox[3] - b.bbox[1])
                proximity = dist / max(ah, bh)
                ma = self._movement_distance(self._history[(camera_id, a.track_id)])
                mb = self._movement_distance(self._history[(camera_id, b.track_id)])
                if proximity <= 1.8 and ma > 18 and mb > 18 and ma + mb > max(width, height) * .18 and self._allow((camera_id, "fight", min(a.track_id, b.track_id), max(a.track_id, b.track_id)), now, 18):
                    conf = min(.95, .58 + .12 * min(1, ma / 70) + .12 * min(1, mb / 70) + .10 * max(0, 1 - proximity / 1.8))
                    events.append(self._new_event(
                        camera_id, now, "FIGHT_SUSPECTED", EventSeverity.HIGH, conf, 84,
                        "Potential physical altercation detected; operator review recommended", "person", a.track_id, None,
                        {"factors": ["two people in close proximity", "rapid movement", "contextual anomaly"],
                         "related_track_ids": [a.track_id, b.track_id], "proximity_ratio": round(proximity, 2),
                         "movement_a": round(ma, 1), "movement_b": round(mb, 1),
                         "note": "Anomaly cue only; operator review recommended."},
                    ))
        return events

    def evaluate(self, camera_id, tracks, zones, frame_shape, timestamp=None):
        now = timestamp or time.time()
        height, width = frame_shape[:2]
        events = []
        current = set()
        night = self._is_night(now)
        persons = [t for t in tracks if t.label == "person"]
        for t in tracks:
            self._history[(camera_id, t.track_id)].append((now, t.center[0], t.center[1]))
        events.extend(self._fight_events(camera_id, persons, now, width, height))

        for zone in zones:
            if not zone.enabled or zone.camera_id != camera_id:
                continue
            persons_inside = 0
            for track in tracks:
                if track.label not in {"person", "car", "truck", "bus", "motorcycle"}:
                    continue
                key = (camera_id, track.track_id, zone.zone_id)
                inside = self._contains_point(zone, *track.center, width, height)
                if inside:
                    current.add(key)
                    persons_inside += track.label == "person"
                    self._entered_at.setdefault(key, now)
                if not inside:
                    self._entered_at.pop(key, None)
                    continue
                history = self._history[(camera_id, track.track_id)]
                direction = self._direction(history)
                dwell = max(0, now - self._entered_at.get(key, now))
                entry = key not in self._inside
                movement = self._movement_distance(history)
                if entry and self._allow(key, now):
                    risk = 82 if zone.zone_type.value == "RESTRICTED" and track.label == "person" else 65
                    factors = ["restricted zone entry"] if zone.zone_type.value == "RESTRICTED" else ["monitored zone entry"]
                    if night and track.label == "person":
                        risk = min(100, risk + 10)
                        factors.append("night-time")
                    if direction and direction != "STATIONARY":
                        factors.append(f"direction {direction.lower()}")
                    severity = EventSeverity.CRITICAL if risk >= 90 else EventSeverity.HIGH if risk >= 75 else EventSeverity.MEDIUM
                    events.append(self._new_event(camera_id, now, "INTRUSION", severity, track.confidence, risk,
                                                  f"{track.label.title()} entered {zone.name}", track.label, track.track_id, zone,
                                                  {"zone_type": zone.zone_type.value, "factors": factors, "night": night,
                                                   "dwell_seconds": round(dwell, 1), "direction": direction}))
                if track.label == "person" and zone.zone_type.value == "RESTRICTED":
                    if dwell >= 30 and self._allow((camera_id, track.track_id, zone.zone_id, "loiter"), now, 15) and movement < max(width, height) * .18:
                        events.append(self._new_event(camera_id, now, "LOITERING", EventSeverity.HIGH, track.confidence, 80,
                                                      f"Person remained in {zone.name} for {int(dwell)}s with limited movement", "person", track.track_id, zone,
                                                      {"dwell_seconds": round(dwell, 1), "movement_pixels": round(movement, 1),
                                                       "factors": ["restricted zone", "extended dwell", "limited movement"]}))
                    if night and self._allow((camera_id, track.track_id, zone.zone_id, "night"), now, 20):
                        events.append(self._new_event(camera_id, now, "NIGHT_MOVEMENT", EventSeverity.HIGH, track.confidence, 75,
                                                      f"Night-time movement detected in {zone.name}", "person", track.track_id, zone,
                                                      {"factors": ["night-time", "restricted zone"], "direction": direction}))
                    if len(history) >= 8 and movement > max(width, height) * .75 and self._allow((camera_id, track.track_id, zone.zone_id, "anomaly"), now, 20):
                        events.append(self._new_event(camera_id, now, "ANOMALY_SUSPECTED", EventSeverity.HIGH, track.confidence, 78,
                                                      f"Unusual movement pattern detected in {zone.name}", "person", track.track_id, zone,
                                                      {"factors": ["unusual movement speed", "restricted zone"],
                                                       "movement_pixels": round(movement, 1), "direction": direction,
                                                       "note": "Operator-assistance signal, not proof of intent."}))
            if persons_inside >= 3 and self._allow((camera_id, zone.zone_id, "group"), now, 20):
                events.append(self._new_event(camera_id, now, "MULTI_PERSON_ACTIVITY", EventSeverity.HIGH, .9, 78,
                                              f"Multiple people detected in {zone.name}", "person", None, zone,
                                              {"person_count": persons_inside, "factors": ["restricted zone", "multiple people"]}))
        self._inside = current
        return events

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if not inter:
            return 0.0
        return inter / max((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter, 1e-6)

    def _assign_drone_tracks(self, camera_id, detections):
        prev = self._drone_tracks[camera_id]
        out = []
        used = set()
        for detection in detections:
            best = None
            score = .2
            for track_id, bbox in prev.items():
                current = self._iou(bbox, detection.bbox)
                if track_id not in used and current > score:
                    best, score = track_id, current
            track_id = best or self._next_drone_id[camera_id]
            self._next_drone_id[camera_id] = max(self._next_drone_id[camera_id], track_id + 1)
            used.add(track_id)
            out.append((track_id, detection))
        self._drone_tracks[camera_id] = {track_id: detection.bbox for track_id, detection in out}
        return out

    def evaluate_drones(self, camera_id, detections, zones, frame_shape, timestamp=None, scan_id=None):
        """Alert only on a classified drone after two specialist scans."""
        now = timestamp or time.time()
        if scan_id is not None and self._last_drone_scan.get(camera_id) == scan_id:
            return []
        if scan_id is not None:
            self._last_drone_scan[camera_id] = scan_id

        height, width = frame_shape[:2]
        drones = [d for d in detections if d.label.strip().lower() == "drone"]
        if not drones:
            self._drone_confirmation[camera_id].clear()
            self._drone_tracks[camera_id] = {}
            return []

        restricted = next((z for z in zones if z.enabled and z.camera_id == camera_id and z.zone_type.value == "RESTRICTED"), None)
        events = []
        current_confirmation = {}
        for track_id, detection in self._assign_drone_tracks(camera_id, drones):
            count = self._drone_confirmation[camera_id].get(track_id, 0) + 1
            current_confirmation[track_id] = count
            if count < 2:
                continue
            x1, y1, x2, y2 = detection.bbox
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            inside = bool(restricted and self._contains_point(restricted, center_x, center_y, width, height))
            if not self._allow((camera_id, "drone", track_id), now, 30):
                continue
            risk = 90 if inside else 70
            severity = EventSeverity.CRITICAL if inside else EventSeverity.HIGH
            factors = ["classified drone", "two-scan confirmation"]
            if inside:
                factors.append("restricted zone")
            message = f"Drone detected in camera {camera_id}"
            if inside:
                message += " — restricted zone"
            events.append(self._new_event(
                camera_id, now, "DRONE_DETECTED", severity, detection.confidence, risk, message,
                "drone", None, restricted if inside else None,
                {"flying_class": "drone", "drone_track_id": track_id, "bbox": list(detection.bbox),
                 "camera_id": camera_id, "detected_at": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                 "factors": factors, "confirmation_scans": count,
                 "note": "Model classification cue; operator review recommended."},
            ))
        self._drone_confirmation[camera_id] = current_confirmation
        return events
