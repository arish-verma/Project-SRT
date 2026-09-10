from __future__ import annotations

import logging
from typing import Any

from app.ai.interfaces import Detection, Detector

logger = logging.getLogger(__name__)


class DroneDetector(Detector):
    """Specialist single-class drone detector using an Ultralytics-compatible weight source."""

    def __init__(self, model_path: str, confidence: float = 0.30) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self._model: Any | None = None
        self._device: str | int = "cpu"

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO
            import torch

            self._device = 0 if torch.cuda.is_available() else "cpu"
            # Hugging Face model repositories are supported explicitly. The
            # fallback keeps local .pt weights working as before.
            if "/" in self.model_path and not self.model_path.lower().endswith((".pt", ".onnx", ".engine")):
                try:
                    self._model = YOLO.from_pretrained(self.model_path)
                except Exception:
                    logger.exception("Could not load drone model via from_pretrained; retrying direct YOLO load")
                    self._model = YOLO(self.model_path)
            else:
                self._model = YOLO(self.model_path)
            self._model.to(self._device)
        return self._model

    def detect(self, frame: Any) -> list[Detection]:
        model = self._load()
        result = model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=640,
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
            detections.append(Detection(
                label="drone" if label.lower() != "drone" else label,
                confidence=float(conf),
                bbox=coords,
            ))
        return detections
