from __future__ import annotations

from typing import Any

from app.ai.interfaces import Detection, Detector


class YOLODetector(Detector):
    """Lazy-loaded Ultralytics detector with explicit CUDA selection when available."""

    def __init__(self, model_path: str = "yolo11n.pt", confidence: float = 0.35) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self._model: Any | None = None
        self._device: str | int = "cpu"

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO
            import torch

            self._device = 0 if torch.cuda.is_available() else "cpu"
            self._model = YOLO(self.model_path)
            self._model.to(self._device)
        return self._model

    @property
    def device(self) -> str | int:
        self._load()
        return self._device

    def detect(self, frame: Any) -> list[Detection]:
        model = self._load()
        result = model.predict(
            source=frame,
            conf=self.confidence,
            device=self._device,
            verbose=False,
        )[0]
        names = result.names
        detections: list[Detection] = []
        if result.boxes is None:
            return detections
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            coords = tuple(float(v) for v in box.tolist())
            label = str(names[int(cls)])
            detections.append(Detection(label=label, confidence=float(conf), bbox=coords))
        return detections
