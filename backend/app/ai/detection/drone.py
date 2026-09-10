from __future__ import annotations

import logging
from typing import Any

from app.ai.interfaces import Detection, Detector

logger = logging.getLogger(__name__)


class DroneDetector(Detector):
    """Specialist single-class drone detector with a tiny-target tiled fallback."""

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

    def _predict(self, frame: Any) -> list[Detection]:
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

    def _tiled_predict(self, frame: Any) -> list[Detection]:
        """Retry on overlapping 2x2 tiles so very small drones occupy more pixels."""
        height, width = frame.shape[:2]
        if width < 160 or height < 160:
            return []

        # 20% overlap prevents a tiny target near a tile boundary from being
        # split out of the receptive field. Each tile is independently resized
        # by YOLO to 640px, effectively giving small aerial targets more scale.
        tile_w = max(160, int(width * 0.60))
        tile_h = max(160, int(height * 0.60))
        x_starts = sorted({0, max(0, width - tile_w)})
        y_starts = sorted({0, max(0, height - tile_h)})

        candidates: list[Detection] = []
        for y0 in y_starts:
            for x0 in x_starts:
                tile = frame[y0:y0 + tile_h, x0:x0 + tile_w]
                for detection in self._predict(tile):
                    x1, y1, x2, y2 = detection.bbox
                    candidates.append(Detection(
                        label=detection.label,
                        confidence=detection.confidence,
                        bbox=(x1 + x0, y1 + y0, x2 + x0, y2 + y0),
                    ))

        # Simple confidence-ordered NMS merges the same drone seen by adjacent
        # overlapping tiles without introducing another dependency.
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        kept: list[Detection] = []
        for candidate in candidates:
            if all(self._iou(candidate.bbox, existing.bbox) < 0.45 for existing in kept):
                kept.append(candidate)
        return kept

    def detect(self, frame: Any) -> list[Detection]:
        detections = self._predict(frame)
        if detections:
            return detections

        # The supplied sample contains a very small aerial target (~20 px wide
        # at 768x432). Full-frame inference can miss targets at this scale, so
        # only pay the extra tiled-inference cost when the normal pass is empty.
        return self._tiled_predict(frame)
