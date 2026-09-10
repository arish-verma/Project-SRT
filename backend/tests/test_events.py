from app.ai.interfaces import Track
from app.schemas.zone import Zone, ZoneType
from app.services.event_engine import RuleEventEngine


def test_restricted_zone_generates_intrusion():
    zone = Zone(zone_id="ZONE-1", camera_id="CAM-1", name="Restricted", zone_type=ZoneType.RESTRICTED,
                polygon=[(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)])
    track = Track(track_id=7, label="person", confidence=0.91, bbox=(10,10,50,100), center=(30,55))
    events = RuleEventEngine().evaluate("CAM-1", [track], [zone], (200,200,3), timestamp=1000)
    assert len(events) == 1
    assert events[0].event_type == "INTRUSION"
    assert events[0].risk_score == 82
    assert events[0].track_id == 7


def test_same_track_is_debounced():
    zone = Zone(zone_id="ZONE-1", camera_id="CAM-1", name="Restricted", zone_type=ZoneType.RESTRICTED,
                polygon=[(0.0,0.0),(1.0,0.0),(1.0,1.0),(0.0,1.0)])
    track = Track(track_id=7, label="person", confidence=0.91, bbox=(10,10,50,100), center=(30,55))
    engine = RuleEventEngine()
    assert len(engine.evaluate("CAM-1", [track], [zone], (200,200,3), timestamp=1000)) == 1
    assert len(engine.evaluate("CAM-1", [track], [zone], (200,200,3), timestamp=1001)) == 0
