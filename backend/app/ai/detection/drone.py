from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

from app.ai.interfaces import Detection, Detector
from app.core.config import settings

logger = logging.getLogger(__name__)
FLYING_LABELS = {"drone", "airplane", "helicopter", "bird"}


class DroneDetector(Detector):
    """Dedicated airborne detector with immediate rendering and temporal alert support."""

    def __init__(self, model_path: str, confidence: float = 0.20) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.model_file = settings.drone_model_file
        self._model: Any | None = None
        self._fallback_model: Any | None = None
        self._device: str | int = "cpu"
        self._model_error: str | None = None
        self._fallback_error: str | None = None
        self._history: dict[str, deque[list[Detection]]] = defaultdict(
            lambda: deque(maxlen=settings.flying_vote_window)
        )
        self._stable: dict[str, list[Detection]] = {}
        self._misses: dict[str, int] = defaultdict(int)

    @staticmethod
    def _canonical_label(label: str) -> str | None:
        value = " ".join(label.strip().lower().replace("_", " ").replace("-", " ").split())
        return {
            "drone": "drone", "uav": "drone", "u a v": "drone",
            "airplane": "airplane", "aircraft": "airplane", "aeroplane": "airplane", "plane": "airplane",
            "helicopter": "helicopter", "bird": "bird",
        }.get(value)

    @staticmethod
    def _threshold(label: str) -> float:
        return {
            "drone": settings.drone_min_confidence,
            "airplane": settings.airplane_min_confidence,
            "helicopter": settings.helicopter_min_confidence,
            "bird": settings.bird_min_confidence,
        }.get(label, 1.0)

    def _downloaded_weights(self, repo_id: str, filename: str) -> str:
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo_id=repo_id, filename=filename)

    def _load_model(self) -> Any:
        from ultralytics import YOLO
        import torch

        self._device = 0 if torch.cuda.is_available() else "cpu"
        try:
            weights = self.model_path if self.model_path.lower().endswith((".pt", ".onnx", ".engine")) else self._downloaded_weights(self.model_path, self.model_file)
            model = YOLO(weights)
            model.to(self._device)
            logger.info("Flying-object primary model loaded: %s on %s", self.model_path, self._device)
            return model
        except Exception as exc:
            self._model_error = str(exc)
            logger.exception("Flying-object primary model failed to load")
            raise

    def _load_fallback(self) -> Any:
        from ultralytics import YOLO
        weights = self._downloaded_weights(settings.drone_fallback_model_path, settings.drone_fallback_model_file)
        model = YOLO(weights)
        model.to(self._device)
        logger.info("Flying-object fallback model loaded: %s", settings.drone_fallback_model_path)
        return model

    def _model_cached(self) -> Any:
        if self._model is None:
            if self._model_error:
                raise RuntimeError(self._model_error)
            self._model = self._load_model()
        return self._model

    def _fallback_cached(self) -> Any | None:
        if self._fallback_model is not None or self._fallback_error:
            return self._fallback_model
        try:
            self._fallback_model = self._load_fallback()
        except Exception as exc:
            self._fallback_error = str(exc)
            logger.warning("Flying-object fallback unavailable: %s", exc)
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

    @staticmethod
    def _center(detection: Detection) -> tuple[float, float]:
        x1, y1, x2, y2 = detection.bbox
        return (x1 + x2) / 2.0, (y1 + y2) / 2.0

    def _predict_model(self, model: Any, frame: Any, confidence: float, fallback: bool = False) -> list[Detection]:
        result = model.predict(source=frame, conf=confidence, imgsz=settings.flying_model_imgsz, device=self._device, verbose=False)[0]
        names = result.names
        out: list[Detection] = []
        if result.boxes is None:
            return out
        h, w = frame.shape[:2]
        frame_area = max(1, w * h)
        for box, conf, cls in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
            label = self._canonical_label(str(names[int(cls)]))
            score = float(conf)
            if label is None or score < self._threshold(label):
                continue
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            bw, bh = x2 - x1, y2 - y1
            if bw < settings.flying_min_box_px or bh < settings.flying_min_box_px:
                continue
            if (max(0.0, bw * bh) / frame_area) > settings.flying_max_box_area_ratio:
                continue
            center_y = ((y1 + y2) / 2.0) / max(1, h)
            if center_y > settings.flying_max_center_y_ratio:
                continue
            out.append(Detection(label=label, confidence=score, bbox=(x1, y1, x2, y2)))
        return self._nms(out)

    def _predict(self, frame: Any) -> list[Detection]:
        try:
            primary = self._predict_model(self._model_cached(), frame, self.confidence)
        except Exception:
            logger.exception("Primary flying inference failed")
            primary = []
        if primary:
            return primary
        fallback = self._fallback_cached()
        if fallback is not None:
            try:
                return self._predict_model(fallback, frame, settings.drone_fallback_confidence, fallback=True)
            except Exception:
                logger.exception("Fallback flying inference failed")
        return []

    def _matching_history(self, camera_id: str, current: Detection) -> list[tuple[int, Detection]]:
        matches: list[tuple[int, Detection]] = []
        for scan_index, scan in enumerate(self._history[camera_id]):
            best = max(scan, key=lambda item: self._iou(current.bbox, item.bbox), default=None)
            if best is not None and self._iou(current.bbox, best.bbox) >= settings.flying_match_iou:
                matches.append((scan_index, best))
        return matches

    def _stable_result(self, camera_id: str, detections: list[Detection], frame_shape: tuple[int, ...]) -> list[Detection]:
        history = self._history[camera_id]
        history.append(detections)
        if not detections:
            self._misses[camera_id] += 1
            if self._misses[camera_id] >= settings.flying_clear_after_misses:
                self._stable[camera_id] = []
                history.clear()
            return list(self._stable.get(camera_id, []))
        self._misses[camera_id] = 0
        height, width = frame_shape[:2]
        min_motion = max(width, height) * settings.flying_min_motion_ratio
        stable: list[Detection] = []
        for current in detections:
            matches = self._matching_history(camera_id, current)
            if len(matches) < settings.flying_required_votes:
                continue
            label_scans: dict[str, list[Detection]] = defaultdict(list)
            for _, item in matches:
                label_scans[item.label].append(item)
            label, votes = max(label_scans.items(), key=lambda item: (len(item[1]), sum(d.confidence for d in item[1])))
            if len(votes) < settings.flying_required_votes or len(votes) / max(1, len(matches)) < settings.flying_label_consensus_ratio:
                continue
            centers = [self._center(item) for _, item in matches]
            displacement = 0.0
            if len(centers) >= 2:
                fx, fy = centers[0]; lx, ly = centers[-1]
                displacement = ((lx - fx) ** 2 + (ly - fy) ** 2) ** 0.5
            avg_conf = sum(d.confidence for d in votes) / len(votes)
            if displacement >= min_motion or (len(votes) >= settings.flying_hover_required_votes and avg_conf >= settings.flying_hover_confidence):
                stable.append(Detection(label=label, confidence=avg_conf, bbox=current.bbox))
        self._stable[camera_id] = self._nms(stable)
        return list(self._stable[camera_id])

    def detect(self, frame: Any, camera_id: str | None = None, allow_tiled: bool = False) -> list[Detection]:
        del allow_tiled
        key = camera_id or "__default__"
        raw = self._predict(frame)
        confirmed = self._stable_result(key, raw, frame.shape)
        return raw if raw else confirmed

    def status(self) -> dict[str, Any]:
        return {
            "primary_model": self.model_path,
            "primary_loaded": self._model is not None,
            "primary_error": self._model_error,
            "fallback_loaded": self._fallback_model is not None,
            "fallback_error": self._fallback_error,
            "device": self._device,
            "classes": sorted(FLYING_LABELS),
            "temporal_votes": settings.flying_required_votes,
            "label_consensus": settings.flying_label_consensus_ratio,
        }

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        kept: list[Detection] = []
        for detection in detections:
            if all(self._iou(detection.bbox, existing.bbox) < 0.45 for existing in kept):
                kept.append(detection)
        return kept
