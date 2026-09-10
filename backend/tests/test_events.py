from datetime import datetime, timezone

from app.ai.interfaces import Track
from app.schemas.zone import Zone, ZoneType
from app.services.event_engine import RuleEventEngine


def test_restricted_zone_generates_intrusion():
    zone = Zone(zone_id="ZONE-1", camera_id="CAM-1", name="Restricted", zone_type=ZoneType.RESTRICTED,
                polygon=[(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)])
    track = Track(track_id=7, label="person", confidence=0.91, bbox=(10,10,50,100), center=(30,55))
    daytime = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc).timestamp()
    events = RuleEventEngine().evaluate("CAM-1", [track], [zone], (200,200,3), timestamp=daytime)
    assert len(events) == 1
    assert events[0].event_type == "INTRUSION"
    assert events[0].risk_score == 82
    assert events[0].track_id == 7


def test_same_track_is_debounced():
    zone = Zone(zone_id="ZONE-1", camera_id="CAM-1", name="Restricted", zone_type=ZoneType.RESTRICTED,
                polygon=[(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)])
    track = Track(track_id=7, label="person", confidence=0.91, bbox=(10,10,50,100), center=(30,55))
    engine = RuleEventEngine()
    daytime = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc).timestamp()
    assert len(engine.evaluate("CAM-1", [track], [zone], (200,200,3), timestamp=daytime)) == 1
    assert len(engine.evaluate("CAM-1", [track], [zone], (200,200,3), timestamp=daytime + 1)) == 0
