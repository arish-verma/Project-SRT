from __future__ import annotations

import logging
from collections import defaultdict, deque
from typing import Any

from app.ai.interfaces import Detection, Detector
from app.core.config import settings

logger = logging.getLogger(__name__)
FLYING_LABELS = {"drone", "airplane", "helicopter", "bird"}


class DroneDetector(Detector):
    """Precision-first airborne-object detector.

    A single frame is never trusted as a live classification. Results are
    accepted only after temporal agreement, and the label is selected by a
    short rolling vote so helicopter/airplane/drone labels cannot oscillate
    frame-to-frame. The independent fallback model is intentionally not used
    for normal inference: mixing models with different class definitions was
    the source of unstable classifications.
    """

    def __init__(self, model_path: str, confidence: float = 0.20) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self.model_file = settings.drone_model_file
        self._model: Any | None = None
        self._device: str | int = "cpu"
        self._model_error: str | None = None
        self._history: dict[str, deque[list[Detection]]] = defaultdict(lambda: deque(maxlen=settings.flying_vote_window))
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

    def _load_model(self) -> Any:
        from ultralytics import YOLO
        import torch

        self._device = 0 if torch.cuda.is_available() else "cpu"
        try:
            if self.model_path.lower().endswith((".pt", ".onnx", ".engine")):
                weights = self.model_path
            else:
                from huggingface_hub import hf_hub_download
                weights = hf_hub_download(repo_id=self.model_path, filename=self.model_file)
            model = YOLO(weights)
            model.to(self._device)
            logger.info("Flying-object model loaded: %s on %s", self.model_path, self._device)
            return model
        except Exception as exc:
            self._model_error = str(exc)
            logger.exception("Flying-object model failed to load")
            raise

    def _model_cached(self) -> Any:
        if self._model is None:
            if self._model_error:
                raise RuntimeError(self._model_error)
            self._model = self._load_model()
        return self._model

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

    def _predict(self, frame: Any) -> list[Detection]:
        model = self._model_cached()
        result = model.predict(
            source=frame,
            conf=self.confidence,
            imgsz=settings.flying_model_imgsz,
            device=self._device,
            verbose=False,
        )[0]
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
            area_ratio = max(0.0, bw * bh) / frame_area
            center_y = (y1 + y2) / 2.0 / max(1, h)
            # Precision gates. Extremely large detections, one-pixel noise and
            # objects glued to the bottom edge are much more likely false
            # positives from ordinary webcam/ground scenes than airborne targets.
            if bw < settings.flying_min_box_px or bh < settings.flying_min_box_px:
                continue
            if area_ratio > settings.flying_max_box_area_ratio:
                continue
            if center_y > settings.flying_max_center_y_ratio:
                continue
            out.append(Detection(label=label, confidence=score, bbox=(x1, y1, x2, y2)))
        return self._nms(out)

    def _stable_result(self, camera_id: str, detections: list[Detection]) -> list[Detection]:
        history = self._history[camera_id]
        history.append(detections)

        if not detections:
            self._misses[camera_id] += 1
            if self._misses[camera_id] >= settings.flying_clear_after_misses:
                self._stable[camera_id] = []
                history.clear()
            return list(self._stable.get(camera_id, []))

        self._misses[camera_id] = 0
        # Match current candidates to recent candidates by IoU, independent of
        # label. This is what prevents the same physical object being renamed
        # when one frame says helicopter and the next says airplane.
        candidates: list[tuple[Detection, list[Detection]]] = []
        for current in detections:
            matches = []
            for scan in history:
                matches.extend(d for d in scan if self._iou(current.bbox, d.bbox) >= settings.flying_match_iou)
            candidates.append((current, matches))

        stable: list[Detection] = []
        required = settings.flying_required_votes
        for current, matches in candidates:
            if len(matches) < required:
                continue
            label_counts: dict[str, list[Detection]] = defaultdict(list)
            for item in matches:
                label_counts[item.label].append(item)
            label, votes = max(label_counts.items(), key=lambda item: (len(item[1]), sum(d.confidence for d in item[1])))
            if len(votes) < required:
                continue
            # Do not switch a stable track on one contrary frame. A new class
            # must win the same temporal vote threshold before replacing it.
            previous = max(
                (d for d in self._stable.get(camera_id, []) if self._iou(current.bbox, d.bbox) >= settings.flying_match_iou),
                key=lambda d: d.confidence,
                default=None,
            )
            if previous is not None and previous.label != label:
                old_votes = len(label_counts.get(previous.label, []))
                if old_votes >= required:
                    label = previous.label
                    votes = label_counts[previous.label]
            avg_conf = sum(d.confidence for d in votes) / len(votes)
            stable.append(Detection(label=label, confidence=avg_conf, bbox=current.bbox))

        self._stable[camera_id] = self._nms(stable)
        return list(self._stable[camera_id])

    def detect(self, frame: Any, camera_id: str | None = None, allow_tiled: bool = False) -> list[Detection]:
        del allow_tiled
        key = camera_id or "__default__"
        try:
            raw = self._predict(frame)
        except Exception:
            logger.exception("Flying-object inference failed for %s", key)
            raw = []
        return self._stable_result(key, raw)

    def status(self) -> dict[str, Any]:
        return {
            "primary_model": self.model_path,
            "primary_loaded": self._model is not None,
            "primary_error": self._model_error,
            "device": self._device,
            "classes": sorted(FLYING_LABELS),
            "precision_mode": True,
            "temporal_votes": settings.flying_required_votes,
        }

    def _nms(self, detections: list[Detection]) -> list[Detection]:
        detections = sorted(detections, key=lambda x: x.confidence, reverse=True)
        kept: list[Detection] = []
        for detection in detections:
            if all(self._iou(detection.bbox, existing.bbox) < .45 for existing in kept):
                kept.append(detection)
        return kept
