from __future__ import annotations

import logging
import threading
from typing import Any

import cv2

from app.ai.interfaces import Detection, Track
from app.ai.detection.drone import DroneDetector
from app.ai.detection.yolo import YOLODetector
from app.ai.tracking.bytetrack import ByteTrackTracker
from app.core.config import settings
from app.services.frame_store import FrameSnapshot, FrameStore

logger = logging.getLogger(__name__)


class AnalyticsPipeline:
    """Shared inference pipeline with camera-isolated tracking state.

    Drone specialist inference is intentionally paused for the current hackathon
    build. The general detector remains fully active, including airplane and other
    aerial classes, so the operator never loses normal object annotations.
    """

    def __init__(self, frame_store: FrameStore, model_path: str = "yolo11n.pt", confidence: float = 0.35) -> None:
        self.frame_store = frame_store
        self.model_path = model_path
        self.confidence = confidence
        self.detector = YOLODetector(model_path, confidence)
        self._trackers: dict[str, ByteTrackTracker] = {}
        self._tracker_lock = threading.RLock()
        # Ultralytics model inference is shared across camera workers. Serialize
        # model calls to keep the shared GPU model stable while camera threads run.
        self._inference_lock = threading.RLock()
        self.drone_detector = DroneDetector(settings.drone_model_path, settings.drone_model_confidence)
        self.drone_scan_interval = max(1, settings.drone_scan_interval)
        self.enabled = True
        self.drone_enabled = False
        self.last_drone_detections: list[Detection] = []
        self._last_drone_error_log = 0.0

    def _tracker_for(self, camera_id: str) -> ByteTrackTracker:
        with self._tracker_lock:
            tracker = self._trackers.get(camera_id)
            if tracker is None:
                tracker = ByteTrackTracker(self.model_path, self.confidence)
                self._trackers[camera_id] = tracker
            return tracker

    def process(self, camera_id: str, packet: Any) -> tuple[list[Detection], list[Track]]:
        frame = packet.frame
        detections: list[Detection] = []
        tracks: list[Track] = []
        self.last_drone_detections = []

        if self.enabled:
            try:
                with self._inference_lock:
                    detections = self.detector.detect(frame)
                    tracks = self._tracker_for(camera_id).update(detections, frame)
            except Exception:
                logger.exception("General analytics inference failed for camera %s", camera_id)
                self.enabled = False

        # Drone detection is deliberately paused. It can be re-enabled later
        # without changing the rest of the event/alert pipeline.
        if self.drone_enabled and packet.frame_index % self.drone_scan_interval == 0:
            try:
                aerial_candidates = [
                    d for d in detections
                    if d.label.lower() in {"airplane", "aircraft", "helicopter"}
                ]
                with self._inference_lock:
                    self.last_drone_detections = self.drone_detector.detect(
                        frame, aerial_candidates=aerial_candidates
                    )
            except Exception as exc:
                self.last_drone_detections = []
                now = __import__("time").time()
                if now - self._last_drone_error_log > 15:
                    logger.error("Drone analytics unavailable for camera %s: %s", camera_id, exc, exc_info=True)
                    self._last_drone_error_log = now

        annotated = frame.copy()
        aerial_labels = {"airplane", "aircraft", "helicopter"}
        for d in detections:
            # When the drone specialist is active it owns aerial display. While
            # paused, the normal YOLO result is displayed like every other class.
            if self.drone_enabled and d.label.lower() in aerial_labels:
                continue
            x1, y1, x2, y2 = map(int, d.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                annotated,
                f"{d.label} {d.confidence:.0%}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        for d in self.last_drone_detections:
            x1, y1, x2, y2 = map(int, d.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                f"DRONE {d.confidence:.0%}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        # Keep track IDs for ordinary detections. Aerial IDs are also useful when
        # the normal detector is in control, but specialist mode hides them.
        for t in tracks:
            if self.drone_enabled and t.label.lower() in aerial_labels:
                continue
            x1, y1, x2, y2 = map(int, t.bbox)
            cv2.putText(
                annotated,
                f"#{t.track_id}",
                (x1, min(annotated.shape[0] - 5, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 82])
        if ok:
            self.frame_store.put(
                camera_id,
                FrameSnapshot(
                    jpeg=encoded.tobytes(),
                    frame_index=packet.frame_index,
                    timestamp=packet.timestamp,
                    detections=len(detections) + len(self.last_drone_detections),
                    tracks=len(tracks),
                ),
            )
        return detections, tracks
