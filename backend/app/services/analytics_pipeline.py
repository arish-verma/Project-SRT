from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from typing import Any

import cv2

from app.ai.detection.drone import DroneDetector
from app.ai.detection.yolo import YOLODetector
from app.ai.interfaces import Detection, Track
from app.ai.tracking.bytetrack import ByteTrackTracker
from app.ai.anpr import ANPREngine
from app.core.config import settings
from app.services.face_service import FaceDetectionService
from app.services.frame_store import FrameSnapshot, FrameStore

logger = logging.getLogger(__name__)
AERIAL_LABELS = {"airplane", "aircraft", "helicopter", "drone"}


@dataclass
class _AerialTrack:
    label: str
    bbox: tuple[float, float, float, float]
    confidence: float
    hits: int
    last_seen: float
    confirmed: bool = False


class AnalyticsPipeline:
    def __init__(self, frame_store: FrameStore, model_path: str = "yolo11n.pt", confidence: float = 0.35) -> None:
        self.frame_store = frame_store
        self.model_path = model_path
        self.confidence = confidence
        self.detector = YOLODetector(model_path, confidence)
        self._trackers: dict[str, ByteTrackTracker] = {}
        self._tracker_lock = threading.RLock()
        # Keep GPU inference serialized on the 4 GB demo GPU. This prevents
        # simultaneous camera workers from exhausting VRAM.
        self._inference_lock = threading.RLock()
        self.drone_detector = DroneDetector(settings.drone_model_path, settings.drone_model_confidence)
        self.drone_scan_interval = max(1, settings.drone_scan_interval)
        self.detection_interval = max(1, settings.detection_interval)
        self.anpr_scan_interval = max(1, settings.anpr_scan_interval)
        self.face_scan_interval = max(1, settings.face_scan_interval)
        self.drone_enabled = True
        self.last_drone_detections: dict[str, list[Detection]] = {}
        self.last_detections: dict[str, list[Detection]] = {}
        self.last_tracks: dict[str, list[Track]] = {}
        self.last_anpr: dict[str, list[dict]] = {}
        self.last_faces: dict[str, list] = {}
        self._aerial_tracks: dict[str, list[_AerialTrack]] = {}
        self.anpr = ANPREngine()
        self.face_service = FaceDetectionService()

    def _tracker_for(self, camera_id: str):
        with self._tracker_lock:
            if camera_id not in self._trackers:
                self._trackers[camera_id] = ByteTrackTracker(self.model_path, self.confidence)
            return self._trackers[camera_id]

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        if inter <= 0:
            return 0.0
        return inter / max(
            max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
            + max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
            - inter,
            1e-6,
        )

    @staticmethod
    def _center_distance(a, b) -> float:
        acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
        bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
        return math.hypot(acx - bcx, acy - bcy)

    def _merge_anpr(self, camera_id: str, new):
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

    def _stabilize_aerial(self, camera_id: str, detections: list[Detection], timestamp: float) -> list[Detection]:
        """Require temporal consistency before exposing a flying-object detection."""
        previous = self._aerial_tracks.get(camera_id, [])
        ttl = settings.aerial_track_ttl_seconds
        active: list[_AerialTrack] = [track for track in previous if timestamp - track.last_seen <= ttl]
        used: set[int] = set()
        updated: list[_AerialTrack] = []

        for detection in detections:
            best_index = None
            best_score = 0.0
            for index, track in enumerate(active):
                if index in used or track.label != detection.label:
                    continue
                iou = self._iou(track.bbox, detection.bbox)
                if iou >= 0.15 and iou > best_score:
                    best_index, best_score = index, iou
                else:
                    # Small aerial targets have unstable IoU. Center-distance
                    # matching keeps a valid track through modest box jitter.
                    area = max(1.0, (track.bbox[2] - track.bbox[0]) * (track.bbox[3] - track.bbox[1]))
                    radius = max(18.0, 0.75 * math.sqrt(area))
                    distance = self._center_distance(track.bbox, detection.bbox)
                    if distance <= radius and best_index is None:
                        best_index = index

            if best_index is None:
                state = _AerialTrack(
                    label=detection.label,
                    bbox=detection.bbox,
                    confidence=detection.confidence,
                    hits=1,
                    last_seen=timestamp,
                )
            else:
                used.add(best_index)
                old = active[best_index]
                state = _AerialTrack(
                    label=detection.label,
                    bbox=detection.bbox,
                    confidence=max(old.confidence * 0.35 + detection.confidence * 0.65, detection.confidence),
                    hits=old.hits + 1,
                    last_seen=timestamp,
                    confirmed=old.confirmed,
                )
            state.confirmed = state.confirmed or state.hits >= settings.aerial_confirmation_hits
            updated.append(state)

        # Keep confirmed tracks alive between specialist scans, but do not let
        # a stale detection generate alerts indefinitely.
        for index, old in enumerate(active):
            if index not in used and old.confirmed:
                updated.append(old)

        self._aerial_tracks[camera_id] = updated
        return [
            Detection(label=track.label, confidence=track.confidence, bbox=track.bbox)
            for track in updated
            if track.confirmed and timestamp - track.last_seen <= ttl
        ]

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
                self.last_faces[camera_id] = []

        if self.drone_enabled and packet.frame_index % self.drone_scan_interval == 0:
            try:
                with self._inference_lock:
                    raw_aerial = self.drone_detector.detect(frame)
                self.last_drone_detections[camera_id] = self._stabilize_aerial(
                    camera_id, raw_aerial, packet.timestamp
                )
            except Exception:
                logger.exception("Aerial inference failed for %s", camera_id)
        elif camera_id not in self.last_drone_detections:
            self.last_drone_detections[camera_id] = []

        aerial_now = self.last_drone_detections.get(camera_id, [])
        annotated = frame.copy()

        for detection in detections:
            if detection.label.lower() in AERIAL_LABELS:
                continue
            x1, y1, x2, y2 = map(int, detection.bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 255, 255), 2)
            cv2.putText(
                annotated,
                f"{detection.label} {detection.confidence:.0%}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        for track in tracks:
            if track.label.lower() in AERIAL_LABELS:
                continue
            x1, y1, x2, y2 = map(int, track.bbox)
            cv2.putText(
                annotated,
                f"#{track.track_id}",
                (x1, min(annotated.shape[0] - 5, y2 + 18)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        for x, y, w, h in self.last_faces.get(camera_id, []):
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (255, 180, 0), 2)
            cv2.putText(annotated, "FACE", (x, max(16, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 180, 0), 1, cv2.LINE_AA)

        for item in self.last_anpr.get(camera_id, []):
            bbox = item.get("plate_bbox") or item.get("bbox")
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 2)
            label = f"PLATE {item['plate_text']}" if item.get("plate_text") else "PLATE CANDIDATE"
            cv2.putText(annotated, label, (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2, cv2.LINE_AA)

        for detection in aerial_now:
            x1, y1, x2, y2 = map(int, detection.bbox)
            label = detection.label.upper()
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(
                annotated,
                f"{label} {detection.confidence:.0%}",
                (x1, max(18, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        ok, encoded = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 76])
        if ok:
            self.frame_store.put(
                camera_id,
                FrameSnapshot(
                    jpeg=encoded.tobytes(),
                    frame_index=packet.frame_index,
                    timestamp=packet.timestamp,
                    detections=len(detections),
                    tracks=len(tracks),
                ),
            )
        return detections, tracks
