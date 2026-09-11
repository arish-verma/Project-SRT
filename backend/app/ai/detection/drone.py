from __future__ import annotations

import logging
from typing import Any

import cv2

from app.ai.interfaces import Detection, Detector
from app.core.config import settings

logger = logging.getLogger(__name__)
FLYING_LABELS = {"drone", "airplane", "helicopter", "bird"}


class DroneDetector(Detector):
    """Strict model-driven detector/classifier for airborne objects.

    The old pipeline normalized every result to ``aerial_object`` and included
    a geometry heuristic. That made class-specific alerts impossible and could
    turn unrelated detections into false aerial targets. This implementation
    accepts only classes emitted by a trained flying-object model.
    """

    def __init__(self, model_path: str, confidence: float = 0.20) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.model_file = settings.drone_model_file
        self.fallback_model_path = settings.drone_fallback_model_path
        self.fallback_model_file = settings.drone_fallback_model_file
        self.fallback_confidence = settings.drone_fallback_confidence
        self._model: Any | None = None
        self._fallback_model: Any | None = None
        self._device: str | int = "cpu"
        self._primary_error: str | None = None
        self._fallback_error: str | None = None

    @staticmethod
    def _canonical_label(label: str) -> str | None:
        value = label.strip().lower().replace("_", " ").replace("-", " ")
        value = " ".join(value.split())
        return {
            "drone": "drone",
            "uav": "drone",
            "u a v": "drone",
            "airplane": "airplane",
            "aircraft": "airplane",
            "aeroplane": "airplane",
            "plane": "airplane",
            "helicopter": "helicopter",
            "bird": "bird",
        }.get(value)

    @staticmethod
    def _threshold(label: str) -> float:
        return {
            "drone": settings.drone_min_confidence,
            "airplane": settings.airplane_min_confidence,
            "helicopter": settings.helicopter_min_confidence,
            "bird": settings.bird_min_confidence,
        }.get(label, 1.0)

    def _load_model(self, model_path: str, filename: str) -> Any:
        from ultralytics import YOLO
        import torch

        self._device = 0 if torch.cuda.is_available() else "cpu"
        if model_path.lower().endswith((".pt", ".onnx", ".engine")):
            weights = model_path
        else:
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(repo_id=model_path, filename=filename)
        model = YOLO(weights)
        model.to(self._device)
        logger.info("Flying-object model loaded: %s (%s) on %s", model_path, filename, self._device)
        return model

    def _load(self) -> Any:
        if self._model is None:
            if self._primary_error:
                raise RuntimeError(self._primary_error)
            try:
                self._model = self._load_model(self.model_path, self.model_file)
            except Exception as exc:
                self._primary_error = str(exc)
                logger.exception("Primary flying-object model failed to load")
                raise
        return self._model

    def _load_fallback(self) -> Any:
        if self._fallback_model is None:
            if self._fallback_error:
                raise RuntimeError(self._fallback_error)
            try:
                self._fallback_model = self._load_model(self.fallback_model_path, self.fallback_model_file)
            except Exception as exc:
                self._fallback_error = str(exc)
                logger.exception("Fallback flying-object model failed to load")
                raise
        return self._fallback_model

    @staticmethod
    def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        ab = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return inter / max(aa + ab - inter, 1e-6)

    def _predict_model(self, model: Any, frame: Any, confidence: float, imgsz: int = 960) -> list[Detection]:
        result = model.predict(source=frame, conf=confidence, imgsz=imgsz, device=self._device, verbose=False)[0]
        names = result.names
        out: list[Detection] = []
        if result.boxes is None:
            return out
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            label = self._canonical_label(str(names[int(cls)]))
            score = float(conf)
            if label is None or score < self._threshold(label):
                continue
            out.append(Detection(label=label, confidence=score, bbox=tuple(float(v) for v in box.tolist())))
        return self._nms(out)

    def _predict(self, frame: Any, imgsz: int = 960) -> list[Detection]:
        return self._predict_model(self._load(), frame, self.confidence, imgsz)

    def _tiled_predict(self, frame: Any, use_fallback: bool = False) -> list[Detection]:
        h, w = frame.shape[:2]
        if w < 160 or h < 160:
            return []
        model = self._load_fallback() if use_fallback else self._load()
        confidence = self.fallback_confidence if use_fallback else self.confidence

        # Two-by-two overlapping tiles. They are a fallback for small distant
        # targets, not the default path, so normal webcam processing stays fast.
        tw, th = max(160, int(w * 0.68)), max(160, int(h * 0.68))
        xs = [0, max(0, w - tw)]
        ys = [0, max(0, h - th)]
        out: list[Detection] = []
        for y0 in sorted(set(ys)):
            for x0 in sorted(set(xs)):
                tile = frame[y0:y0 + th, x0:x0 + tw]
                for d in self._predict_model(model, tile, confidence, 960):
                    x1, y1, x2, y2 = d.bbox
                    out.append(Detection(label=d.label, confidence=d.confidence,
                                         bbox=(x1 + x0, y1 + y0, x2 + x0, y2 + y0)))
        return self._nms(out)

    def detect(self, frame: Any, camera_id: str | None = None) -> list[Detection]:
        """Classify airborne objects without heuristic guesses."""
        del camera_id

        # Primary model: Javvanny YOLOv8m. It explicitly distinguishes
        # Drone/Airplane/Helicopter/Bird.
        try:
            detections = self._predict(frame, imgsz=960)
            if detections:
                return detections
            # Small-target rescue only after a clean full-frame miss.
            try:
                tiled = self._tiled_predict(frame, use_fallback=False)
                if tiled:
                    return tiled
            except Exception:
                logger.exception("Primary tiled flying-object inference failed")
            return []
        except Exception:
            logger.exception("Primary flying-object inference failed; using fallback")

        # Fallback: AeroYOLO distinguishes aircraft/drone/helicopter.
        try:
            detections = self._predict_model(self._load_fallback(), frame, self.fallback_confidence, 960)
            if detections:
                return detections
            return self._tiled_predict(frame, use_fallback=True)
        except Exception:
            logger.exception("Fallback flying-object inference failed")
            return []

    def status(self) -> dict[str, Any]:
        return {
            "primary_model": self.model_path,
            "primary_loaded": self._model is not None,
            "primary_error": self._primary_error,
            "fallback_model": self.fallback_model_path,
            "fallback_loaded": self._fallback_model is not None,
            "fallback_error": self._fallback_error,
            "device": self._device,
            "classes": sorted(FLYING_LABELS),
        }

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        kept: list[Detection] = []
        for detection in detections:
            if all(detection.label != existing.label or self._iou(detection.bbox, existing.bbox) < .45 for existing in kept):
                kept.append(detection)
        return kept
