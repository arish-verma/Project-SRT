from app.ai.interfaces import Track
from app.services.event_engine import RuleEventEngine


def person(track_id: int, x: float, y: float = 100.0) -> Track:
    return Track(track_id=track_id, label="person", confidence=0.9,
                 bbox=(x - 20, y - 50, x + 20, y + 50), center=(x, y))


def test_close_bilateral_motion_triggers_fight():
    engine = RuleEventEngine()
    samples = [(100.0, 300.0), (112.0, 288.0), (126.0, 304.0),
               (142.0, 286.0), (158.0, 310.0), (174.0, 292.0)]
    events = []
    for index, (xa, xb) in enumerate(samples):
        events.extend(engine.evaluate(
            "CAM-FIGHT", [person(1, xa), person(2, xb)], [],
            (720, 1280, 3), timestamp=1000.0 + index * 0.1,
        ))
    assert any(event.event_type == "FIGHT_SUSPECTED" for event in events)
