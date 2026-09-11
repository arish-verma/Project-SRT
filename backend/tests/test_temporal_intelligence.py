from datetime import datetime, timezone

from app.ai.interfaces import Detection, Track
from app.schemas.zone import Zone, ZoneType
from app.services.event_engine import RuleEventEngine


def zone():
    return Zone(zone_id="ZONE-1", camera_id="CAM-1", name="Restricted Border Area", zone_type=ZoneType.RESTRICTED,
                polygon=[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)])


def track(x: float, y: float = 50.0):
    return Track(track_id=7, label="person", confidence=0.9, bbox=(x - 10, y - 20, x + 10, y + 20), center=(x, y))


def test_loitering_after_extended_dwell():
    engine = RuleEventEngine()
    z = zone()
    start = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc).timestamp()
    assert any(e.event_type == "INTRUSION" for e in engine.evaluate("CAM-1", [track(50)], [z], (100, 100, 3), start))
    events = engine.evaluate("CAM-1", [track(51)], [z], (100, 100, 3), start + 31)
    assert any(e.event_type == "LOITERING" for e in events)


def test_night_movement_is_flagged():
    engine = RuleEventEngine()
    z = zone()
    night = datetime(2026, 9, 10, 23, 30, tzinfo=timezone.utc).timestamp()
    events = engine.evaluate("CAM-1", [track(50)], [z], (100, 100, 3), night)
    assert any(e.event_type == "NIGHT_MOVEMENT" for e in events)


def test_drone_gets_stable_track_id_and_restricted_risk():
    engine = RuleEventEngine()
    z = zone()
    detection = Detection(label="drone", confidence=0.9, bbox=(40, 40, 60, 60))
    now = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc).timestamp()

    assert engine.evaluate_drones("CAM-1", [detection], [z], (100, 100, 3), now, scan_id=1) == []
    events = engine.evaluate_drones("CAM-1", [detection], [z], (100, 100, 3), now + 1, scan_id=2)

    assert len(events) == 1
    assert events[0].event_type == "DRONE_DETECTED"
    assert events[0].risk_score == 90
    assert events[0].metadata["drone_track_id"] == 1
