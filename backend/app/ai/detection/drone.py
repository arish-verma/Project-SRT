from __future__ import annotations

import logging
from typing import Any

from app.ai.interfaces import Detection, Detector
from app.core.config import settings

logger = logging.getLogger(__name__)

# AeroYOLO is explicitly trained for these three classes.
PRIMARY_CLASS_MAP = {
    "aircraft": "airplane",
    "airplane": "airplane",
    "plane": "airplane",
    "drone": "drone",
    "uav": "drone",
    "quadcopter": "drone",
    "helicopter": "helicopter",
    "heli": "helicopter",
}
SUPPORTED_AERIAL_CLASSES = {"airplane", "drone", "helicopter"}


class DroneDetector(Detector):
    """Dedicated flying-object detector.

    The primary AeroYOLO model returns the real class instead of collapsing every
    target into ``aerial_object``.  The single-class fallback is used only when
    the primary classifier cannot be loaded, and therefore can only return
    ``drone``.  Generic person/vehicle detections are never promoted to aerial
    targets.
    """

    def __init__(self, model_path: str, confidence: float = 0.25) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.fallback_model_path = settings.drone_fallback_model_path
        self.fallback_confidence = settings.drone_fallback_confidence
        self._model: Any | None = None
        self._fallback_model: Any | None = None
        self._device: str | int = "cpu"
        self._primary_failed = False
        self._fallback_failed = False

    def _load_model(self, model_path: str):
        from ultralytics import YOLO
        import torch

        self._device = 0 if torch.cuda.is_available() else "cpu"
        if "/" in model_path and not model_path.lower().endswith((".pt", ".onnx", ".engine")):
            from huggingface_hub import hf_hub_download

            weights = hf_hub_download(repo_id=model_path, filename="best.pt")
            model = YOLO(weights)
        else:
            model = YOLO(model_path)
        model.to(self._device)
        logger.info("Aerial model loaded: %s on %s", model_path, self._device)
        return model

    def _load(self):
        if self._model is None:
            if self._primary_failed:
                return None
            try:
                self._model = self._load_model(self.model_path)
            except Exception:
                self._primary_failed = True
                logger.exception("Unable to load primary aerial classifier: %s", self.model_path)
        return self._model

    def _load_fallback(self):
        if self._fallback_model is None:
            if self._fallback_failed:
                return None
            try:
                self._fallback_model = self._load_model(self.fallback_model_path)
            except Exception:
                self._fallback_failed = True
                logger.exception("Unable to load fallback drone detector: %s", self.fallback_model_path)
        return self._fallback_model

    @staticmethod
    def _normalize_primary_label(label: str) -> str | None:
        return PRIMARY_CLASS_MAP.get(label.strip().lower())

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

    @staticmethod
    def _geometry_ok_for_drone(frame_shape, bbox) -> bool:
        """Reject implausibly large 'drone' boxes such as a webcam face."""
        height, width = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        bw = max(0.0, x2 - x1)
        bh = max(0.0, y2 - y1)
        if width <= 0 or height <= 0 or bw <= 0 or bh <= 0:
            return False
        area_ratio = (bw * bh) / float(width * height)
        return (
            area_ratio <= settings.drone_max_area_ratio
            and bw / width <= settings.drone_max_width_ratio
            and bh / height <= settings.drone_max_height_ratio
        )

    def _predict_primary(self, frame, confidence: float, imgsz: int) -> list[Detection]:
        model = self._load()
        if model is None:
            return []
        result = model.predict(
            source=frame,
            conf=confidence,
            imgsz=imgsz,
            device=self._device,
            half=self._device != "cpu",
            verbose=False,
        )[0]
        names = result.names
        out: list[Detection] = []
        if result.boxes is None:
            return out
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            label = self._normalize_primary_label(str(names[int(cls)]))
            if label is None:
                continue
            bbox = tuple(float(v) for v in box.tolist())
            if label == "drone" and not self._geometry_ok_for_drone(frame.shape, bbox):
                continue
            out.append(Detection(label=label, confidence=float(conf), bbox=bbox))
        return self._nms(out)

    def _predict_fallback(self, frame, confidence: float, imgsz: int) -> list[Detection]:
        model = self._load_fallback()
        if model is None:
            return []
        result = model.predict(
            source=frame,
            conf=confidence,
            imgsz=imgsz,
            device=self._device,
            half=self._device != "cpu",
            verbose=False,
        )[0]
        out: list[Detection] = []
        if result.boxes is None:
            return out
        for box, conf in zip(result.boxes.xyxy, result.boxes.conf):
            bbox = tuple(float(v) for v in box.tolist())
            if not self._geometry_ok_for_drone(frame.shape, bbox):
                continue
            out.append(Detection(label="drone", confidence=float(conf), bbox=bbox))
        return self._nms(out)

    def _tiled_primary(self, frame) -> list[Detection]:
        height, width = frame.shape[:2]
        if width < 160 or height < 160:
            return []
        tile_w = max(160, int(width * 0.70))
        tile_h = max(160, int(height * 0.70))
        output: list[Detection] = []
        xs = sorted({0, max(0, width - tile_w)})
        ys = sorted({0, max(0, height - tile_h)})
        for y0 in ys:
            for x0 in xs:
                tile = frame[y0:y0 + tile_h, x0:x0 + tile_w]
                for detection in self._predict_primary(tile, settings.aerial_tiled_confidence, 960):
                    x1, y1, x2, y2 = detection.bbox
                    output.append(
                        Detection(
                            label=detection.label,
                            confidence=detection.confidence,
                            bbox=(x1 + x0, y1 + y0, x2 + x0, y2 + y0),
                        )
                    )
        return self._nms(output)

    def _tiled_fallback(self, frame) -> list[Detection]:
        height, width = frame.shape[:2]
        if width < 160 or height < 160:
            return []
        tile_w = max(160, int(width * 0.70))
        tile_h = max(160, int(height * 0.70))
        output: list[Detection] = []
        xs = sorted({0, max(0, width - tile_w)})
        ys = sorted({0, max(0, height - tile_h)})
        for y0 in ys:
            for x0 in xs:
                tile = frame[y0:y0 + tile_h, x0:x0 + tile_w]
                for detection in self._predict_fallback(tile, self.fallback_confidence, 960):
                    x1, y1, x2, y2 = detection.bbox
                    output.append(
                        Detection(
                            label="drone",
                            confidence=detection.confidence,
                            bbox=(x1 + x0, y1 + y0, x2 + x0, y2 + y0),
                        )
                    )
        return self._nms(output)

    def detect(self, frame, aerial_candidates=None) -> list[Detection]:
        """Detect and classify actual flying objects.

        ``aerial_candidates`` is intentionally ignored.  A COCO person/vehicle
        detector is not a valid aerial classifier and must never be used to
        manufacture an aerial target.
        """
        del aerial_candidates
        try:
            primary = self._predict_primary(frame, self.confidence, settings.aerial_image_size)
            if primary:
                return primary
            tiled = self._tiled_primary(frame)
            if tiled:
                return tiled
        except Exception:
            logger.exception("Primary aerial inference failed; using drone-only fallback")

        # The fallback is intentionally used only when the primary model is
        # unavailable/failed. It cannot label aircraft or helicopters as drones.
        try:
            fallback = self._predict_fallback(frame, self.fallback_confidence, settings.aerial_image_size)
            if fallback:
                return fallback
            return self._tiled_fallback(frame)
        except Exception:
            logger.exception("Fallback drone inference failed")
            return []

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        detections.sort(key=lambda item: item.confidence, reverse=True)
        kept: list[Detection] = []
        for detection in detections:
            if all(self._iou(detection.bbox, existing.bbox) < 0.45 for existing in kept):
                kept.append(detection)
        return kept
