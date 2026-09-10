from __future__ import annotations

import logging
from typing import Any

import cv2

from app.ai.interfaces import Detection, Detector

logger = logging.getLogger(__name__)


class DroneDetector(Detector):
    """Specialist aerial detector with tiny-target candidate refinement.

    AeroYOLO distinguishes aircraft, drone and helicopter. General YOLO can
    localize a tiny drone as ``airplane``; those aerial candidates are cropped,
    enlarged and classified by AeroYOLO. We never map airplane directly to
    drone.
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
                # Download the actual best.pt explicitly. This avoids model
                # loader/version ambiguity with Hugging Face repositories.
                from huggingface_hub import hf_hub_download
                model_file = hf_hub_download(repo_id=self.model_path, filename="best.pt")
                self._model = YOLO(model_file)
            else:
                self._model = YOLO(self.model_path)
            self._model.to(self._device)
            logger.info("Drone model loaded: %s", self.model_path)
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
            if label != "drone":
                continue
            coords = tuple(float(v) for v in box.tolist())
            detections.append(Detection(label="drone", confidence=float(conf), bbox=coords))
        return detections

    def _candidate_predict(self, frame: Any, candidates: list[Detection]) -> list[Detection]:
        """Classify localized aerial candidates at several enlarged scales."""
        height, width = frame.shape[:2]
        results: list[Detection] = []
        for candidate in candidates:
            if candidate.label.lower() not in {"airplane", "aircraft", "helicopter"}:
                continue

            x1, y1, x2, y2 = candidate.bbox
            bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
            # For a ~20 px target, keep enough sky context while making the
            # target occupy a useful fraction of the specialist input.
            for pad_factor in (1.5, 2.5, 4.0):
                pad = max(16.0, max(bw, bh) * pad_factor)
                cx1, cy1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
                cx2, cy2 = min(width, int(x2 + pad)), min(height, int(y2 + pad))
                crop = frame[cy1:cy2, cx1:cx2]
                if crop.size == 0:
                    continue

                target_size = 960
                scale = max(1.0, target_size / max(crop.shape[:2]))
                if scale > 1.0:
                    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

                for detection in self._predict(crop, confidence=0.05, imgsz=960):
                    dx1, dy1, dx2, dy2 = detection.bbox
                    inv = 1.0 / scale
                    mapped = (dx1 * inv + cx1, dy1 * inv + cy1, dx2 * inv + cx1, dy2 * inv + cy1)
                    results.append(Detection(label="drone", confidence=detection.confidence, bbox=mapped))

        results.sort(key=lambda item: item.confidence, reverse=True)
        kept: list[Detection] = []
        for result in results:
            if all(self._iou(result.bbox, existing.bbox) < 0.45 for existing in kept):
                kept.append(result)
        return kept

    def _tiled_predict(self, frame: Any) -> list[Detection]:
        """Run overlapping higher-resolution tiles for tiny aerial targets."""
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
                for detection in self._predict(tile, confidence=0.06, imgsz=960):
                    x1, y1, x2, y2 = detection.bbox
                    candidates.append(Detection(label="drone", confidence=detection.confidence, bbox=(x1 + x0, y1 + y0, x2 + x0, y2 + y0)))
        candidates.sort(key=lambda item: item.confidence, reverse=True)
        kept: list[Detection] = []
        for candidate in candidates:
            if all(self._iou(candidate.bbox, existing.bbox) < 0.45 for existing in kept):
                kept.append(candidate)
        return kept

    def detect(self, frame: Any, aerial_candidates: list[Detection] | None = None) -> list[Detection]:
        # Candidate refinement is intentionally first: a tiny drone that fills
        # only a few dozen pixels in the full frame is much easier to classify
        # after localization and enlargement.
        if aerial_candidates:
            candidates = self._candidate_predict(frame, aerial_candidates)
            if candidates:
                return candidates

        detections = self._predict(frame, imgsz=960)
        if detections:
            return detections
        return self._tiled_predict(frame)
