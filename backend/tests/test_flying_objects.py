from app.ai.detection.drone import DroneDetector
from app.ai.interfaces import Detection
from app.services.event_engine import RuleEventEngine


def test_flying_model_label_normalization():
    assert DroneDetector._canonical_label("Drone") == "drone"
    assert DroneDetector._canonical_label("Airplane") == "airplane"
    assert DroneDetector._canonical_label("aircraft") == "airplane"
    assert DroneDetector._canonical_label("Helicopter") == "helicopter"
    assert DroneDetector._canonical_label("Bird") == "bird"
    assert DroneDetector._canonical_label("person") is None


def test_only_drone_generates_drone_alert():
    engine = RuleEventEngine()
    detections = [
        Detection(label="airplane", confidence=.95, bbox=(20, 20, 100, 70)),
        Detection(label="helicopter", confidence=.90, bbox=(120, 20, 190, 80)),
        Detection(label="bird", confidence=.80, bbox=(210, 20, 240, 45)),
    ]
    assert engine.evaluate_drones("CAM-1", detections, [], (300, 400, 3), timestamp=100.0, scan_id=1) == []


def test_drone_requires_two_specialist_scans_before_alert():
    engine = RuleEventEngine()
    drone = Detection(label="drone", confidence=.88, bbox=(150, 50, 220, 90))

    first = engine.evaluate_drones("CAM-1", [drone], [], (300, 400, 3), timestamp=100.0, scan_id=1)
    second = engine.evaluate_drones("CAM-1", [drone], [], (300, 400, 3), timestamp=101.0, scan_id=2)

    assert first == []
    assert len(second) == 1
    assert second[0].event_type == "DRONE_DETECTED"
    assert second[0].object_type == "drone"
    assert second[0].metadata["flying_class"] == "drone"
