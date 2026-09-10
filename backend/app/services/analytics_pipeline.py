from __future__ import annotations

import logging
from typing import Any

import cv2

from app.ai.interfaces import Detection, Track
from app.ai.detection.yolo import YOLODetector
from app.ai.tracking.bytetrack import ByteTrackTracker
from app.services.frame_store import FrameSnapshot, FrameStore

logger = logging.getLogger(__name__)


class AnalyticsPipeline:
    """Per-frame perception/tracking pipeline with lazy model initialization."""

    def __init__(self, frame_store: FrameStore, model_path: str = "yolo11n.pt", confidence: float = 0.35) -> None:
        self.frame_store = frame_store
        self.model_path = model_path
        self.confidence = confidence
        self.detector = YOLODetector(model_path, confidence)
        self.tracker = ByteTrackTracker(model_path, confidence)
        self.enabled = True

    def process(self, camera_id: str, packet: Any) -> tuple[list[Detection], list[Track]]:
        frame = packet.frame
        detections: list[Detection] = []
        tracks: list[Track] = []
        if self.enabled:
            try:
                detections = self.detector.detect(frame)
                tracks = self.tracker.update(detections, frame)
            except Exception:
                # Keep video streaming even if optional CV dependencies/weights are unavailable.
                logger.exception("Analytics inference failed for camera %s", camera_id)
                self.enabled = False

        annotated = frame.copy()
        for d in detections:
            x1, y1, x2, y2 = map(int, d.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(annotated, f"{d.label} {d.confidence:.0%}", (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        for t in tracks:
            x1, y1, x2, y2 = map(int, t.bbox)
            cv2.putText(annotated, f"#{t.track_id}", (x1, min(annotated.shape[0] - 5, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            self.frame_store.put(camera_id, FrameSnapshot(
                jpeg=encoded.tobytes(), frame_index=packet.frame_index,
                timestamp=packet.timestamp, detections=len(detections), tracks=len(tracks),
            ))
        return detections, tracks
