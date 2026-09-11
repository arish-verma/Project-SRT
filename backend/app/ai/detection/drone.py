from __future__ import annotations

import logging
from typing import Any

import cv2

from app.ai.interfaces import Detection, Detector
from app.core.config import settings

logger = logging.getLogger(__name__)


class DroneDetector(Detector):
    """Aerial detector with a multi-class model and RGB drone fallback.

    The primary AeroYOLO model distinguishes aircraft, drone and helicopter.
    The RGB drone-only model is used for localized aerial candidates and as a
    tiled fallback for tiny targets that are difficult to classify at full-frame
    resolution. Generic airplane predictions are never directly relabeled as drones.
    """

    def __init__(self, model_path: str, confidence: float = 0.20) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.fallback_model_path = settings.drone_fallback_model_path
        self.fallback_confidence = settings.drone_fallback_confidence
        self._model: Any | None = None
        self._fallback_model: Any | None = None
        self._device: str | int = "cpu"

    def _load_model(self, model_path: str) -> Any:
        from ultralytics import YOLO
        import torch

        self._device = 0 if torch.cuda.is_available() else "cpu"
        if "/" in model_path and not model_path.lower().endswith((".pt", ".onnx", ".engine")):
            from huggingface_hub import hf_hub_download
            model_file = hf_hub_download(repo_id=model_path, filename="best.pt")
            model = YOLO(model_file)
        else:
            model = YOLO(model_path)
        model.to(self._device)
        logger.info("Drone model loaded: %s", model_path)
        return model

    def _load(self) -> Any:
        if self._model is None:
            self._model = self._load_model(self.model_path)
        return self._model

    def _load_fallback(self) -> Any:
        if self._fallback_model is None:
            self._fallback_model = self._load_model(self.fallback_model_path)
        return self._fallback_model

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

    def _predict_model(
        self,
        model: Any,
        frame: Any,
        confidence: float,
        imgsz: int = 960,
    ) -> list[Detection]:
        result = model.predict(
            source=frame,
            conf=confidence,
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

    def _predict(self, frame: Any, confidence: float | None = None, imgsz: int = 960) -> list[Detection]:
        return self._predict_model(
            self._load(),
            frame,
            self.confidence if confidence is None else confidence,
            imgsz,
        )

    @staticmethod
    def _crop(frame: Any, candidate: Detection, pad_factor: float) -> tuple[Any, float, int, int] | None:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = candidate.bbox
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        pad = max(16.0, max(bw, bh) * pad_factor)
        cx1, cy1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
        cx2, cy2 = min(width, int(x2 + pad)), min(height, int(y2 + pad))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None

        target_size = 960
        scale = max(1.0, target_size / max(crop.shape[:2]))
        if scale > 1.0:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return crop, scale, cx1, cy1

    def _candidate_predict(self, frame: Any, candidates: list[Detection]) -> list[Detection]:
        """Classify localized aerial candidates with AeroYOLO first."""
        results: list[Detection] = []
        for candidate in candidates:
            if candidate.label.lower() not in {"airplane", "aircraft", "helicopter"}:
                continue

            for pad_factor in (1.5, 2.5, 4.0):
                prepared = self._crop(frame, candidate, pad_factor)
                if prepared is None:
                    continue
                crop, scale, cx1, cy1 = prepared
                for detection in self._predict(crop, confidence=0.05, imgsz=960):
                    dx1, dy1, dx2, dy2 = detection.bbox
                    inv = 1.0 / scale
                    mapped = (dx1 * inv + cx1, dy1 * inv + cy1, dx2 * inv + cx1, dy2 * inv + cy1)
                    results.append(Detection(label="drone", confidence=detection.confidence, bbox=mapped))

        return self._nms(results)

    def _fallback_candidate_predict(self, frame: Any, candidates: list[Detection]) -> list[Detection]:
        """Use the drone-only RGB model on localized aerial candidates."""
        results: list[Detection] = []
        model = self._load_fallback()

        for candidate in candidates:
            if candidate.label.lower() not in {"airplane", "aircraft", "helicopter"}:
                continue

            # Enlarging the candidate gives a tiny 20-30 px drone considerably
            # more pixels without scanning arbitrary unrelated image regions.
            for pad_factor in (2.5, 4.0):
                prepared = self._crop(frame, candidate, pad_factor)
                if prepared is None:
                    continue
                crop, scale, cx1, cy1 = prepared
                detections = self._predict_model(
                    model,
                    crop,
                    confidence=self.fallback_confidence,
                    imgsz=960,
                )
                for detection in detections:
                    dx1, dy1, dx2, dy2 = detection.bbox
                    inv = 1.0 / scale
                    mapped = (dx1 * inv + cx1, dy1 * inv + cy1, dx2 * inv + cx1, dy2 * inv + cy1)
                    results.append(Detection(label="drone", confidence=detection.confidence, bbox=mapped))
                if results:
                    break

        return self._nms(results)

    def _tiled_fallback_predict(self, frame: Any) -> list[Detection]:
        """Run the RGB drone specialist on overlapping tiles for tiny targets.

        This is the final specialist path. It is still a dedicated drone model,
        not a relabeling of COCO airplane detections, so the operator only sees
        DRONE when a drone-specific model actually returns a detection.
        """
        height, width = frame.shape[:2]
        if width < 160 or height < 160:
            return []

        model = self._load_fallback()
        tile_w = max(160, int(width * 0.60))
        tile_h = max(160, int(height * 0.60))
        x_starts = sorted({0, max(0, width - tile_w)})
        y_starts = sorted({0, max(0, height - tile_h)})
        results: list[Detection] = []

        for y0 in y_starts:
            for x0 in x_starts:
                tile = frame[y0:y0 + tile_h, x0:x0 + tile_w]
                if tile.size == 0:
                    continue
                detections = self._predict_model(
                    model,
                    tile,
                    confidence=self.fallback_confidence,
                    imgsz=960,
                )
                for detection in detections:
                    x1, y1, x2, y2 = detection.bbox
                    results.append(Detection(
                        label="drone",
                        confidence=detection.confidence,
                        bbox=(x1 + x0, y1 + y0, x2 + x0, y2 + y0),
                    ))

        return self._nms(results)

    def _tiled_predict(self, frame: Any) -> list[Detection]:
        """Run overlapping AeroYOLO tiles for tiny aerial targets."""
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
        return self._nms(candidates)

    def detect(self, frame: Any, aerial_candidates: list[Detection] | None = None) -> list[Detection]:
        # 1. Refine general YOLO aerial candidates with the multi-class model.
        if aerial_candidates:
            primary = self._candidate_predict(frame, aerial_candidates)
            if primary:
                return primary

            # 2. Dedicated RGB drone model on the same localized evidence.
            fallback = self._fallback_candidate_predict(frame, aerial_candidates)
            if fallback:
                logger.info("Tiny aerial candidate classified by RGB drone fallback")
                return fallback

        # 3. Try AeroYOLO directly for larger/cleaner drone views.
        detections = self._predict(frame, imgsz=960)
        if detections:
            return detections
        tiled = self._tiled_predict(frame)
        if tiled:
            return tiled

        # 4. Last resort for tiny RGB drones: the dedicated drone-only model on
        # overlapping tiles. This is what catches targets that neither full-frame
        # specialist inference nor the generic aerial candidate can resolve.
        fallback_tiled = self._tiled_fallback_predict(frame)
        if fallback_tiled:
            logger.info("Tiny drone detected by tiled RGB fallback")
        return fallback_tiled

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        detections.sort(key=lambda item: item.confidence, reverse=True)
        kept: list[Detection] = []
        for detection in detections:
            if all(self._iou(detection.bbox, existing.bbox) < 0.45 for existing in kept):
                kept.append(detection)
        return kept
