from __future__ import annotations

from typing import Any

from app.ai.interfaces import Detection, Detector


class YOLODetector(Detector):
    """Lazy-loaded Ultralytics detector. Keeps CV dependency out of API startup/tests."""

    def __init__(self, model_path: str = "yolo11n.pt", confidence: float = 0.35) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self._model: Any | None = None

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        return self._model

    def detect(self, frame: Any) -> list[Detection]:
        model = self._load()
        result = model.predict(source=frame, conf=self.confidence, verbose=False)[0]
        names = result.names
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            coords = tuple(float(v) for v in box.tolist())
            label = str(names[int(cls)])
            detections.append(Detection(label=label, confidence=float(conf), bbox=coords))
        return detections
