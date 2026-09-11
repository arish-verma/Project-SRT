from types import SimpleNamespace

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


def test_flying_detector_does_not_expose_single_frame_candidate():
    detector = DroneDetector("unused", confidence=.25)
    candidate = Detection(label="helicopter", confidence=.90, bbox=(120, 60, 180, 110))
    detector._predict = lambda frame: [candidate]
    frame = SimpleNamespace(shape=(360, 360, 3))

    assert detector.detect(frame, camera_id="CAM-1") == []


def test_flying_detector_locks_persistent_moving_class():
    detector = DroneDetector("unused", confidence=.25)
    boxes = [
        (80, 55, 135, 100),
        (92, 58, 147, 103),
        (106, 62, 161, 107),
    ]
    detections = [Detection(label="helicopter", confidence=.80, bbox=b) for b in boxes]
    index = {"value": 0}

    def predict(frame):
        value = detections[min(index["value"], len(detections) - 1)]
        index["value"] += 1
        return [value]

    detector._predict = predict
    frame = SimpleNamespace(shape=(360, 360, 3))

    assert detector.detect(frame, camera_id="CAM-1") == []
    assert detector.detect(frame, camera_id="CAM-1") == []
    locked = detector.detect(frame, camera_id="CAM-1")

    assert len(locked) == 1
    assert locked[0].label == "helicopter"


def test_flying_detector_resists_one_frame_class_flip():
    detector = DroneDetector("unused", confidence=.25)
    sequence = [
        Detection(label="helicopter", confidence=.80, bbox=(80, 55, 135, 100)),
        Detection(label="helicopter", confidence=.82, bbox=(92, 58, 147, 103)),
        Detection(label="helicopter", confidence=.84, bbox=(106, 62, 161, 107)),
        Detection(label="airplane", confidence=.95, bbox=(118, 65, 173, 110)),
    ]
    index = {"value": 0}

    def predict(frame):
        value = sequence[min(index["value"], len(sequence) - 1)]
        index["value"] += 1
        return [value]

    detector._predict = predict
    frame = SimpleNamespace(shape=(360, 360, 3))

    detector.detect(frame, camera_id="CAM-2")
    detector.detect(frame, camera_id="CAM-2")
    locked = detector.detect(frame, camera_id="CAM-2")
    assert locked and locked[0].label == "helicopter"

    after_flip = detector.detect(frame, camera_id="CAM-2")
    assert after_flip and after_flip[0].label == "helicopter"


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
