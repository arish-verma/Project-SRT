from __future__ import annotations

from typing import Any

from app.ai.interfaces import Detection, Track, Tracker


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


class ByteTrackTracker(Tracker):
    """Lightweight IoU tracker fallback with stable IDs.

    The interface is deliberately named after the intended tracker boundary; a full
    ByteTrack implementation can replace this class without changing the pipeline.
    """

    def __init__(self, model_path: str = "", confidence: float = 0.35, iou_threshold: float = 0.25) -> None:
        self.iou_threshold = iou_threshold
        self._next_id = 1
        self._previous: dict[int, tuple[str, tuple[float, float, float, float]]] = {}

    def update(self, detections: list[Detection], frame: Any) -> list[Track]:
        del frame
        candidates = dict(self._previous)
        result: list[Track] = []
        used: set[int] = set()
        for detection in detections:
            best_id = None
            best_score = self.iou_threshold
            for track_id, (label, bbox) in candidates.items():
                if track_id in used or label != detection.label:
                    continue
                score = _iou(bbox, detection.bbox)
                if score > best_score:
                    best_score, best_id = score, track_id
            if best_id is None:
                best_id = self._next_id
                self._next_id += 1
            used.add(best_id)
            self._previous[best_id] = (detection.label, detection.bbox)
            x1, y1, x2, y2 = detection.bbox
            result.append(Track(
                track_id=best_id, label=detection.label, confidence=detection.confidence,
                bbox=detection.bbox, center=((x1 + x2) / 2, (y1 + y2) / 2),
            ))
        # Forget tracks not observed in this frame.
        self._previous = {k: v for k, v in self._previous.items() if k in used}
        return result
