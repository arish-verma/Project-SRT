from datetime import datetime, timezone

from app.schemas.alert import AlertStatus
from app.schemas.event import EventRecord, EventSeverity
from app.services.alert_store import AlertStore
from app.services.risk_engine import RiskEngine
from app.ai.interfaces import Track


def test_risk_engine_is_bounded_and_explainable():
    track = Track(track_id=1, label="person", confidence=0.9, bbox=(0, 0, 10, 10), center=(5, 5))
    engine = RiskEngine()
    assert engine.score(track, restricted=True, night=True, dwell_seconds=31) == 75
    assert engine.severity(75) == EventSeverity.HIGH


def test_alert_store_persists(tmp_path):
    store = AlertStore(str(tmp_path / "events.db"))
    event = EventRecord(event_id="EVT-TEST", camera_id="CAM-1", timestamp=datetime.now(timezone.utc),
                        event_type="INTRUSION", severity=EventSeverity.HIGH, confidence=.9,
                        risk_score=80, object_type="person", message="Person entered restricted zone")
    alert = store.ensure_for_event(event)
    assert alert.status == AlertStatus.NEW
    assert store.list()[0].event_id == event.event_id
    assert store.update(alert.alert_id, AlertStatus.ACKNOWLEDGED).status == AlertStatus.ACKNOWLEDGED
