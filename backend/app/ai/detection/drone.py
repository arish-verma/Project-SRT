from __future__ import annotations

import logging
from typing import Any

from app.ai.interfaces import Detection, Detector

logger = logging.getLogger(__name__)


class DroneDetector(Detector):
    """Specialist aerial detector with a small-target tiled fallback.

    The configured model is multi-class (aircraft / drone / helicopter), so
    only an explicit ``drone`` prediction is promoted to SRT's drone event.
    Generic aircraft predictions are deliberately ignored here.
    """

    def __init__(self, model_path: str, confidence: float = 0.20) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self._model: Any | None = None
        self._device: str | int = "cpu"

    def _load(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO
            import torch

            self._device = 0 if torch.cuda.is_available() else "cpu"
            if "/" in self.model_path and not self.model_path.lower().endswith((".pt", ".onnx", ".engine")):
                try:
                    self._model = YOLO.from_pretrained(self.model_path)
                except Exception:
                    logger.exception("Could not load aerial model via from_pretrained; retrying direct YOLO load")
                    self._model = YOLO(self.model_path)
            else:
                self._model = YOLO(self.model_path)
            self._model.to(self._device)
        return self._model

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        if inter <= 0:
            return 0.0
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def _predict(self, frame: Any, confidence: float | None = None, imgsz: int = 960) -> list[Detection]:
        model = self._load()
        result = model.predict(
            source=frame,
            conf=self.confidence if confidence is None else confidence,
            imgsz=imgsz,
            device=self._device,
            verbose=False,
        )[0]
        names = result.names
        detections: list[Detection] = []
        if result.boxes is None:
            return detections

        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            label = str(names[int(cls)]).strip().lower()
            # Critical safety/accuracy rule: never convert aircraft into drone.
            if label != "drone":
                continue
            coords = tuple(float(v) for v in box.tolist())
            detections.append(Detection(
                label="drone",
                confidence=float(conf),
                bbox=coords,
            ))
        return detections

    def _tiled_predict(self, frame: Any) -> list[Detection]:
        """Run higher-resolution overlapping tiles for tiny aerial targets."""
        height, width = frame.shape[:2]
        if width < 160 or height < 160:
            return []

        tile_w = max(160, int(width * 0.60))
        tile_h = max(160, int(height * 0.60))
        x_starts = sorted({0, max(0, width - tile_w)})
        y_starts = sorted({0, max(0, height - tile_h)})

        candidates: list[Detection] = []
        for y0 in y_starts:
            for x0 in x_starts:
                tile = frame[y0:y0 + tile_h, x0:x0 + tile_w]
                for detection in self._predict(tile, confidence=max(0.12, self.confidence * 0.75), imgsz=960):
                    x1, y1, x2, y2 = detection.bbox
                    candidates.append(Detection(
                        label="drone",
                        confidence=detection.confidence,
                        bbox=(x1 + x0, y1 + y0, x2 + x0, y2 + y0),
                    ))

        candidates.sort(key=lambda item: item.confidence, reverse=True)
        kept: list[Detection] = []
        for candidate in candidates:
            if all(self._iou(candidate.bbox, existing.bbox) < 0.45 for existing in kept):
                kept.append(candidate)
        return kept

    def detect(self, frame: Any) -> list[Detection]:
        detections = self._predict(frame, imgsz=960)
        if detections:
            return detections
        return self._tiled_predict(frame)
