from __future__ import annotations
import logging
from typing import Any
import cv2
from app.ai.interfaces import Detection, Detector
from app.core.config import settings

logger = logging.getLogger(__name__)
AERIAL_LABELS = {'drone', 'airplane', 'aircraft', 'helicopter', 'aerial_object'}


class DroneDetector(Detector):
    """Aerial-target detector with resilient primary/fallback inference."""

    def __init__(self, model_path: str, confidence: float = .10) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.fallback_model_path = settings.drone_fallback_model_path
        self.fallback_confidence = settings.drone_fallback_confidence
        self._model = None
        self._fallback_model = None
        self._device = 'cpu'

    def _load_model(self, model_path):
        from ultralytics import YOLO
        import torch
        self._device = 0 if torch.cuda.is_available() else 'cpu'
        if '/' in model_path and not model_path.lower().endswith(('.pt', '.onnx', '.engine')):
            from huggingface_hub import hf_hub_download
            weights = hf_hub_download(repo_id=model_path, filename='best.pt')
            model = YOLO(weights)
        else:
            model = YOLO(model_path)
        model.to(self._device)
        logger.info('Aerial model loaded: %s on %s', model_path, self._device)
        return model

    def _load(self):
        if self._model is None:
            self._model = self._load_model(self.model_path)
        return self._model

    def _load_fallback(self):
        if self._fallback_model is None:
            self._fallback_model = self._load_model(self.fallback_model_path)
        return self._fallback_model

    @staticmethod
    def _iou(a, b):
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        ab = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / max(aa + ab - inter, 1e-6)

    @staticmethod
    def _aerial(label: str) -> bool:
        return label.strip().lower() in AERIAL_LABELS

    def _predict_model(self, model, frame, confidence, imgsz=1280):
        result = model.predict(
            source=frame,
            conf=confidence,
            imgsz=imgsz,
            device=self._device,
            verbose=False,
        )[0]
        names = result.names
        out = []
        if result.boxes is None:
            return out
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            label = str(names[int(cls)]).strip().lower()
            if not self._aerial(label):
                continue
            out.append(Detection(
                label='aerial_object',
                confidence=float(conf),
                bbox=tuple(float(v) for v in box.tolist()),
            ))
        return out

    def _predict(self, frame, confidence=None, imgsz=1280):
        return self._predict_model(
            self._load(), frame,
            self.confidence if confidence is None else confidence,
            imgsz,
        )

    @staticmethod
    def _crop(frame, candidate, pad_factor):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = candidate.bbox
        bw, bh = max(1., x2 - x1), max(1., y2 - y1)
        pad = max(16., max(bw, bh) * pad_factor)
        cx1, cy1 = max(0, int(x1 - pad)), max(0, int(y1 - pad))
        cx2, cy2 = min(width, int(x2 + pad)), min(height, int(y2 + pad))
        crop = frame[cy1:cy2, cx1:cx2]
        if crop.size == 0:
            return None
        scale = max(1., 960 / max(crop.shape[:2]))
        if scale > 1:
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        return crop, scale, cx1, cy1

    def _candidate_predict(self, frame, candidates):
        out = []
        for c in candidates:
            if not self._aerial(c.label):
                continue
            for pad in (1.5, 2.5, 4.0):
                p = self._crop(frame, c, pad)
                if p is None:
                    continue
                crop, scale, cx1, cy1 = p
                for d in self._predict(crop, confidence=.05, imgsz=960):
                    x1, y1, x2, y2 = d.bbox
                    inv = 1 / scale
                    out.append(Detection(
                        label='aerial_object', confidence=d.confidence,
                        bbox=(x1 * inv + cx1, y1 * inv + cy1, x2 * inv + cx1, y2 * inv + cy1),
                    ))
        return self._nms(out)

    def _fallback_candidate_predict(self, frame, candidates):
        out = []
        model = self._load_fallback()
        for c in candidates:
            if not self._aerial(c.label):
                continue
            for pad in (2.5, 4.0):
                p = self._crop(frame, c, pad)
                if p is None:
                    continue
                crop, scale, cx1, cy1 = p
                for d in self._predict_model(model, crop, self.fallback_confidence, 960):
                    x1, y1, x2, y2 = d.bbox
                    inv = 1 / scale
                    out.append(Detection(
                        label='aerial_object', confidence=d.confidence,
                        bbox=(x1 * inv + cx1, y1 * inv + cy1, x2 * inv + cx1, y2 * inv + cy1),
                    ))
                if out:
                    break
        return self._nms(out)

    def _tiled_predict(self, frame):
        h, w = frame.shape[:2]
        if w < 160 or h < 160:
            return []
        out = []
        tw, th = max(160, int(w * .65)), max(160, int(h * .65))
        for y0 in sorted({0, max(0, h - th)}):
            for x0 in sorted({0, max(0, w - tw)}):
                tile = frame[y0:y0 + th, x0:x0 + tw]
                for d in self._predict(tile, confidence=.045, imgsz=960):
                    x1, y1, x2, y2 = d.bbox
                    out.append(Detection(
                        label='aerial_object', confidence=d.confidence,
                        bbox=(x1 + x0, y1 + y0, x2 + x0, y2 + y0),
                    ))
        return self._nms(out)

    def _tiled_fallback_predict(self, frame):
        h, w = frame.shape[:2]
        if w < 160 or h < 160:
            return []
        model = self._load_fallback()
        tw, th = max(160, int(w * .65)), max(160, int(h * .65))
        out = []
        for y0 in sorted({0, max(0, h - th)}):
            for x0 in sorted({0, max(0, w - tw)}):
                tile = frame[y0:y0 + th, x0:x0 + tw]
                for d in self._predict_model(model, tile, self.fallback_confidence, 960):
                    x1, y1, x2, y2 = d.bbox
                    out.append(Detection(
                        label='aerial_object', confidence=d.confidence,
                        bbox=(x1 + x0, y1 + y0, x2 + x0, y2 + y0),
                    ))
        return self._nms(out)

    def _heuristic_aerial(self, frame, candidates):
        h, w = frame.shape[:2]
        out = []
        for c in candidates:
            if not self._aerial(c.label):
                continue
            x1, y1, x2, y2 = c.bbox
            bw, bh = max(1, x2 - x1), max(1, y2 - y1)
            area = bw * bh / (w * h)
            cy = (y1 + y2) / 2
            if area <= .02 and cy < h * .82 and .25 <= bw / bh <= 4.0:
                out.append(Detection(
                    label='aerial_object',
                    confidence=min(.88, max(.52, float(c.confidence) * .85)),
                    bbox=c.bbox,
                ))
        return self._nms(out)

    def detect(self, frame, aerial_candidates=None):
        # Never let a primary model/download failure disable the entire aerial pipeline.
        if aerial_candidates:
            try:
                primary = self._candidate_predict(frame, aerial_candidates)
                if primary:
                    return primary
            except Exception:
                logger.exception('Primary aerial candidate inference failed; trying fallback')
            try:
                fallback = self._fallback_candidate_predict(frame, aerial_candidates)
                if fallback:
                    return fallback
            except Exception:
                logger.exception('Fallback aerial candidate inference failed')
            heuristic = self._heuristic_aerial(frame, aerial_candidates)
            if heuristic:
                return heuristic

        try:
            detections = self._predict(frame, imgsz=1280)
            if detections:
                return detections
        except Exception:
            logger.exception('Primary aerial full-frame inference failed; trying fallback')

        try:
            tiled = self._tiled_predict(frame)
            if tiled:
                return tiled
        except Exception:
            logger.exception('Primary aerial tiled inference failed; trying fallback')

        try:
            return self._tiled_fallback_predict(frame)
        except Exception:
            logger.exception('Fallback aerial inference failed')
            return []

    def _nms(self, detections):
        detections.sort(key=lambda x: x.confidence, reverse=True)
        kept = []
        for d in detections:
            if all(self._iou(d.bbox, e.bbox) < .45 for e in kept):
                kept.append(d)
        return kept
