from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
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
    """Real-time pipeline plus a deterministic synchronous path for uploaded clips."""

    def __init__(self, frame_store: FrameStore, model_path: str = "yolo11n.pt", confidence: float = 0.35) -> None:
        self.frame_store = frame_store
        self.model_path = model_path
        self.confidence = confidence
        self.detector = YOLODetector(model_path, confidence)
        self._trackers: dict[str, ByteTrackTracker] = {}
        self._tracker_lock = threading.RLock()
        self._base_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="srt-base-ai")
        self._flying_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="srt-flying-ai")
        self._aux_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="srt-aux-ai")
        self._state_lock = threading.RLock()
        self._base_busy: set[str] = set(); self._flying_busy: set[str] = set(); self._anpr_busy: set[str] = set(); self._face_busy: set[str] = set()
        self._latest_frames: dict[str, tuple[Any, int, float]] = {}
        self.drone_detector = DroneDetector(settings.drone_model_path, settings.drone_model_confidence)
        self.drone_scan_interval = max(1, settings.drone_scan_interval)
        self.drone_tiled_scan_every = max(1, settings.drone_tiled_scan_every)
        self.detection_interval = max(1, settings.detection_interval)
        self.anpr_scan_interval = max(1, settings.anpr_scan_interval)
        self.face_scan_interval = max(1, settings.face_scan_interval)
        self.drone_enabled = True
        self.last_detections: dict[str, list[Detection]] = {}
        self.last_tracks: dict[str, list[Any]] = {}
        self.last_flying_detections: dict[str, list[Detection]] = {}
        self.last_drone_detections = self.last_flying_detections
        self.last_flying_scan_frame: dict[str, int] = {}
        self._flying_scan_count: dict[str, int] = {}
        self.last_anpr: dict[str, list[dict]] = {}
        self.last_faces: dict[str, list[tuple[int, int, int, int]]] = {}
        self.anpr = ANPREngine(); self.face_service = FaceDetectionService()

    def _tracker_for(self, camera_id: str) -> ByteTrackTracker:
        with self._tracker_lock:
            if camera_id not in self._trackers:
                self._trackers[camera_id] = ByteTrackTracker(self.model_path, self.confidence)
            return self._trackers[camera_id]

    @staticmethod
    def _iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a; bx1, by1, bx2, by2 = b
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        if inter <= 0: return 0.0
        aa = max(0, ax2 - ax1) * max(0, ay2 - ay1); ab = max(0, bx2 - bx1) * max(0, by2 - by1)
        return inter / max(aa + ab - inter, 1e-6)

    def _filter_airborne_overlaps(self, detections: list[Detection], flying: list[Detection]) -> list[Detection]:
        """Prevent a helicopter/aircraft box from being presented as a person."""
        if not flying: return detections
        out = []
        for d in detections:
            if d.label.lower() == "person":
                x1, y1, x2, y2 = d.bbox; cx = (x1 + x2) / 2; cy = (y1 + y2) / 2
                covered = any(self._iou(d.bbox, f.bbox) >= 0.12 or (f.bbox[0] <= cx <= f.bbox[2] and f.bbox[1] <= cy <= f.bbox[3]) for f in flying)
                if covered: continue
            out.append(d)
        return out

    def _merge_anpr(self, camera_id: str, new: list[dict]) -> list[dict]:
        old = self.last_anpr.get(camera_id, []); merged = []
        for item in new:
            best = max((p for p in old if p.get("vehicle_type") == item.get("vehicle_type")), key=lambda p: self._iou(item["bbox"], p.get("bbox", [0,0,0,0])), default=None)
            if best and self._iou(item["bbox"], best.get("bbox", [0,0,0,0])) >= .15 and not item.get("plate_text") and best.get("plate_text"):
                item = dict(item); item["plate_text"] = best["plate_text"]; item["plate_bbox"] = best.get("plate_bbox"); item["status"] = "READ"
            merged.append(item)
        return merged

    def _annotate_and_store(self, camera_id: str, frame: Any, frame_index: int, timestamp: float, detections, tracks) -> None:
        annotated = frame.copy()
        for detection in detections:
            if detection.label.lower() in FLYING_LABELS: continue
            x1,y1,x2,y2 = map(int, detection.bbox); cv2.rectangle(annotated,(x1,y1),(x2,y2),(255,255,255),2)
            cv2.putText(annotated,f"{detection.label} {detection.confidence:.0%}",(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        for track in tracks:
            if track.label.lower() in FLYING_LABELS: continue
            x1,y1,x2,y2 = map(int,track.bbox); cv2.putText(annotated,f"#{track.track_id}",(x1,min(annotated.shape[0]-5,y2+18)),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA)
        for x,y,w,h in self.last_faces.get(camera_id,[]):
            cv2.rectangle(annotated,(x,y),(x+w,y+h),(255,180,0),2); cv2.putText(annotated,"FACE",(x,max(16,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,180,0),1,cv2.LINE_AA)
        for plate in self.last_anpr.get(camera_id,[]):
            bbox=plate.get("plate_bbox") or plate.get("bbox"); x1,y1,x2,y2=map(int,bbox); cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,255),2)
            cv2.putText(annotated,f"PLATE {plate['plate_text']}" if plate.get('plate_text') else "PLATE CANDIDATE",(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2,cv2.LINE_AA)
        for d in self.last_flying_detections.get(camera_id,[]):
            x1,y1,x2,y2=map(int,d.bbox); cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,0,255),2); cv2.putText(annotated,f"{d.label.upper()} {d.confidence:.0%}",(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2,cv2.LINE_AA)
        ok, encoded=cv2.imencode('.jpg',annotated,[cv2.IMWRITE_JPEG_QUALITY,76])
        if ok: self.frame_store.put(camera_id,FrameSnapshot(jpeg=encoded.tobytes(),frame_index=frame_index,timestamp=timestamp,detections=len(detections),tracks=len(tracks)))

    def _republish_latest(self,camera_id:str)->None:
        with self._state_lock:
            latest=self._latest_frames.get(camera_id); detections=list(self.last_detections.get(camera_id,[])); tracks=list(self.last_tracks.get(camera_id,[]))
        if latest is not None:
            frame,idx,ts=latest; self._annotate_and_store(camera_id,frame,idx,ts,detections,tracks)

    def infer_sync(self,camera_id:str,frame:Any,frame_index:int,timestamp:float):
        """Run base + airborne inference before returning; used for finite local clips."""
        flying = self.drone_detector.detect(frame,camera_id=camera_id,allow_tiled=True) if self.drone_enabled else []
        detections = self._filter_airborne_overlaps(self.detector.detect(frame), flying)
        tracks = self._tracker_for(camera_id).update(detections,frame)
        with self._state_lock:
            self.last_detections[camera_id]=detections; self.last_tracks[camera_id]=tracks
            self.last_flying_detections[camera_id]=flying; self.last_flying_scan_frame[camera_id]=frame_index
            self._latest_frames[camera_id]=(frame.copy(),frame_index,timestamp)
        self._annotate_and_store(camera_id,frame,frame_index,timestamp,detections,tracks)
        return detections,tracks,flying

    def _run_base_inference(self,camera_id,frame):
        try:
            detections=self.detector.detect(frame)
            flying=self.last_flying_detections.get(camera_id,[])
            detections=self._filter_airborne_overlaps(detections,flying)
            tracks=self._tracker_for(camera_id).update(detections,frame)
            with self._state_lock: self.last_detections[camera_id]=detections; self.last_tracks[camera_id]=tracks
            self._republish_latest(camera_id)
        except Exception: logger.exception("Analytics inference failed for %s",camera_id)
        finally:
            with self._state_lock:self._base_busy.discard(camera_id)

    def _schedule_base(self,camera_id,frame):
        with self._state_lock:
            if camera_id in self._base_busy:return
            self._base_busy.add(camera_id)
        self._base_executor.submit(self._run_base_inference,camera_id,frame.copy())

    def _run_flying_inference(self,camera_id,frame,scan_frame,scan_count):
        try:
            flying=self.drone_detector.detect(frame,camera_id=camera_id,allow_tiled=scan_count%self.drone_tiled_scan_every==0)
            with self._state_lock:self.last_flying_detections[camera_id]=flying;self.last_flying_scan_frame[camera_id]=scan_frame
            self._republish_latest(camera_id)
        except Exception:
            logger.exception("Flying-object inference failed for %s",camera_id)
            with self._state_lock:self.last_flying_detections[camera_id]=[];self.last_flying_scan_frame[camera_id]=scan_frame
            self._republish_latest(camera_id)
        finally:
            with self._state_lock:self._flying_busy.discard(camera_id)

    def _schedule_flying(self,camera_id,frame,frame_index):
        with self._state_lock:
            if camera_id in self._flying_busy:return
            n=self._flying_scan_count.get(camera_id,0)+1;self._flying_scan_count[camera_id]=n;self._flying_busy.add(camera_id)
        self._flying_executor.submit(self._run_flying_inference,camera_id,frame.copy(),frame_index,n)

    def _run_anpr(self,camera_id,frame,detections,timestamp):
        try:self.last_anpr[camera_id]=self._merge_anpr(camera_id,self.anpr.scan(frame,detections,camera_id=camera_id,timestamp=timestamp));self._republish_latest(camera_id)
        except Exception:logger.exception("ANPR failed for %s",camera_id)
        finally:
            with self._state_lock:self._anpr_busy.discard(camera_id)

    def _schedule_anpr(self,camera_id,frame,detections,timestamp):
        with self._state_lock:
            if camera_id in self._anpr_busy:return
            self._anpr_busy.add(camera_id)
        self._aux_executor.submit(self._run_anpr,camera_id,frame.copy(),list(detections),timestamp)

    def _run_face(self,camera_id,frame):
        try:self.last_faces[camera_id]=self.face_service.detect(frame);self._republish_latest(camera_id)
        except Exception:logger.exception("Face detection failed for %s",camera_id)
        finally:
            with self._state_lock:self._face_busy.discard(camera_id)

    def _schedule_face(self,camera_id,frame):
        with self._state_lock:
            if camera_id in self._face_busy:return
            self._face_busy.add(camera_id)
        self._aux_executor.submit(self._run_face,camera_id,frame.copy())

    def process(self,camera_id,packet):
        frame=packet.frame
        with self._state_lock:
            self._latest_frames[camera_id]=(frame.copy(),packet.frame_index,packet.timestamp);detections=list(self.last_detections.get(camera_id,[]));tracks=list(self.last_tracks.get(camera_id,[]))
        self._annotate_and_store(camera_id,frame,packet.frame_index,packet.timestamp,detections,tracks)
        if packet.frame_index%self.detection_interval==0:self._schedule_base(camera_id,frame)
        if packet.frame_index%self.anpr_scan_interval==0:self._schedule_anpr(camera_id,frame,detections,packet.timestamp)
        if packet.frame_index%self.face_scan_interval==0:self._schedule_face(camera_id,frame)
        if self.drone_enabled and packet.frame_index%self.drone_scan_interval==0:self._schedule_flying(camera_id,frame,packet.frame_index)
        return detections,tracks

    def shutdown(self):
        for executor in (self._base_executor,self._flying_executor,self._aux_executor):executor.shutdown(wait=False,cancel_futures=True)
