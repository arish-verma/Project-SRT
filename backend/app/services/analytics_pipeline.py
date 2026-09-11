from __future__ import annotations

import logging
import threading
from typing import Any

import cv2

from app.ai.detection.drone import DroneDetector, FLYING_LABELS
from app.ai.detection.yolo import YOLODetector
from app.ai.interfaces import Detection
from app.ai.tracking.bytetrack import ByteTrackTracker
from app.ai.anpr import ANPREngine
from app.core.config import settings
from app.services.face_service import FaceDetectionService
from app.services.frame_store import FrameSnapshot, FrameStore

logger = logging.getLogger(__name__)


class AnalyticsPipeline:
    """Runs the real-time perception stack while keeping frame delivery responsive."""

    def __init__(self, frame_store: FrameStore, model_path: str = "yolo11n.pt", confidence: float = 0.35) -> None:
        self.frame_store = frame_store
        self.model_path = model_path
        self.confidence = confidence
        self.detector = YOLODetector(model_path, confidence)
        self._trackers: dict[str, ByteTrackTracker] = {}
        self._tracker_lock = threading.RLock()
        # GPU inference is intentionally serialized on the 4 GB laptop GPU.
        self._inference_lock = threading.RLock()
        self.drone_detector = DroneDetector(settings.drone_model_path, settings.drone_model_confidence)
        self.drone_scan_interval = max(1, settings.drone_scan_interval)
        self.detection_interval = max(1, settings.detection_interval)
        self.anpr_scan_interval = max(1, settings.anpr_scan_interval)
        self.face_scan_interval = max(1, settings.face_scan_interval)
        self.drone_enabled = True
        # Historical name retained for API compatibility. Values are now
        # classified flying objects rather than generic aerial_object boxes.
        self.last_drone_detections: dict[str, list[Detection]] = {}
        self.last_flying_detections = self.last_drone_detections
        self.last_flying_scan_frame: dict[str, int] = {}
        self.last_detections: dict[str, list[Detection]] = {}
        self.last_tracks = {}
        self.last_anpr: dict[str, list[dict]] = {}
        self.last_faces: dict[str, list[tuple[int, int, int, int]]] = {}
        self.anpr = ANPREngine()
        self.face_service = FaceDetectionService()

    def _tracker_for(self, camera_id: str) -> ByteTrackTracker:
        with self._tracker_lock:
            if camera_id not in self._trackers:
                self._trackers[camera_id] = ByteTrackTracker(self.model_path, self.confidence)
            return self._trackers[camera_id]

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        aa = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        ab = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / max(aa + ab - inter, 1e-6)

    def _merge_anpr(self, camera_id: str, new: list[dict]) -> list[dict]:
        old = self.last_anpr.get(camera_id, [])
        merged = []
        for item in new:
            best = max(
                (p for p in old if p.get("vehicle_type") == item.get("vehicle_type")),
                key=lambda p: self._iou(item["bbox"], p.get("bbox", [0, 0, 0, 0])),
                default=None,
            )
            if best and self._iou(item["bbox"], best.get("bbox", [0, 0, 0, 0])) >= 0.15 and not item.get("plate_text") and best.get("plate_text"):
                item = dict(item)
                item["plate_text"] = best["plate_text"]
                item["plate_bbox"] = best.get("plate_bbox")
                item["status"] = "READ"
            merged.append(item)
        return merged

    def _annotate_and_store(self, camera_id: str, frame: Any, frame_index: int, timestamp: float, detections, tracks) -> None:
        """Publish the live frame before optional heavy specialist work."""
        annotated = frame.copy()

        for detection in detections:
            if detection.label.lower() in FLYING_LABELS:
                continue
            x1, y1, x2, y2 = map(int, detection.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(annotated, f"{detection.label} {detection.confidence:.0%}",
                        (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .5,
                        (255, 255, 255), 1, cv2.LINE_AA)

        for track in tracks:
            if track.label.lower() in FLYING_LABELS:
                continue
            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.putText(annotated, f"#{track.track_id}",
                        (x1, min(annotated.shape[0] - 5, y2 + 18)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2, cv2.LINE_AA)

        for x, y, w, h in self.last_faces.get(camera_id, []):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 180, 0), 2)
            cv2.putText(annotated, "FACE", (x, max(16, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 180, 0), 1, cv2.LINE_AA)

        for plate in self.last_anpr.get(camera_id, []):
            bbox = plate.get("plate_bbox") or plate.get("bbox")
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"PLATE {plate['plate_text']}" if plate.get("plate_text") else "PLATE CANDIDATE"
            cv2.putText(annotated, label, (x1, max(18, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 255, 255), 2, cv2.LINE_AA)

        for detection in self.last_flying_detections.get(camera_id, []):
            x1, y1, x2, y2 = map(int, detection.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(annotated, f"{detection.label.upper()} {detection.confidence:.0%}",
                        (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .55,
                        (0, 0, 255), 2, cv2.LINE_AA)

        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 76])
        if ok:
            self.frame_store.put(
                camera_id,
                FrameSnapshot(jpeg=encoded.tobytes(), frame_index=frame_index,
                              timestamp=timestamp, detections=len(detections), tracks=len(tracks)),
            )

    def process(self, camera_id: str, packet: Any):
        frame = packet.frame
        detections = self.last_detections.get(camera_id, [])
        tracks = self.last_tracks.get(camera_id, [])

        if packet.frame_index % self.detection_interval == 0:
            try:
                with self._inference_lock:
                    detections = self.detector.detect(frame)
                    tracks = self._tracker_for(camera_id).update(detections, frame)
            except Exception:
                logger.exception("Analytics inference failed for %s", camera_id)

        self.last_detections[camera_id] = detections
        self.last_tracks[camera_id] = tracks

        # Publish before ANPR/face/flying-object specialist inference.
        self._annotate_and_store(camera_id, frame, packet.frame_index, packet.timestamp, detections, tracks)

        if packet.frame_index % self.anpr_scan_interval == 0:
            try:
                self.last_anpr[camera_id] = self._merge_anpr(
                    camera_id,
                    self.anpr.scan(frame, detections, camera_id=camera_id, timestamp=packet.timestamp),
                )
            except Exception:
                logger.exception("ANPR failed for %s", camera_id)
                self.last_anpr[camera_id] = []

        if packet.frame_index % self.face_scan_interval == 0:
            try:
                self.last_faces[camera_id] = self.face_service.detect(frame)
            except Exception:
                logger.exception("Face detection failed for %s", camera_id)
                self.last_faces[camera_id] = []

        if self.drone_enabled and packet.frame_index % self.drone_scan_interval == 0:
            self.last_flying_scan_frame[camera_id] = packet.frame_index
            try:
                with self._inference_lock:
                    flying = self.drone_detector.detect(frame, camera_id=camera_id)
                self.last_flying_detections[camera_id] = flying
            except Exception:
                logger.exception("Flying-object inference failed for %s", camera_id)
                self.last_flying_detections[camera_id] = []

        return detections, tracks

    @staticmethod
    def _dedupe_aerial(items):
        """Backward-compatible helper for older integrations."""
        out = []
        for detection in sorted(items, key=lambda x: x.confidence, reverse=True):
            if all(detection.label != existing.label or AnalyticsPipeline._iou(detection.bbox, existing.bbox) < .45 for existing in out):
                out.append(detection)
        return out
