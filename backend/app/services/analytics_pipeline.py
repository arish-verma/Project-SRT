from __future__ import annotations
import logging,threading
from typing import Any
import cv2
from app.ai.interfaces import Detection,Track
from app.ai.detection.drone import DroneDetector
from app.ai.detection.yolo import YOLODetector
from app.ai.tracking.bytetrack import ByteTrackTracker
from app.ai.anpr import ANPREngine
from app.core.config import settings
from app.services.face_service import FaceDetectionService
from app.services.frame_store import FrameSnapshot,FrameStore
logger=logging.getLogger(__name__)
AERIAL_LABELS={'airplane','aircraft','helicopter','drone','aerial_object'}

class AnalyticsPipeline:
    def __init__(self,frame_store:FrameStore,model_path:str='yolo11n.pt',confidence:float=.35)->None:
        self.frame_store=frame_store;self.model_path=model_path;self.confidence=confidence;self.detector=YOLODetector(model_path,confidence);self._trackers={};self._tracker_lock=threading.RLock();self._inference_lock=threading.RLock();self.drone_detector=DroneDetector(settings.drone_model_path,settings.drone_model_confidence);self.drone_scan_interval=max(1,settings.drone_scan_interval);self.detection_interval=max(1,settings.detection_interval);self.anpr_scan_interval=max(1,settings.anpr_scan_interval);self.face_scan_interval=max(1,settings.face_scan_interval);self.drone_enabled=True;self.last_drone_detections={};self.last_detections={};self.last_tracks={};self.last_anpr={};self.last_faces={};self.anpr=ANPREngine();self.face_service=FaceDetectionService()
    def _tracker_for(self,camera_id:str):
        with self._tracker_lock:
            if camera_id not in self._trackers:self._trackers[camera_id]=ByteTrackTracker(self.model_path,self.confidence)
            return self._trackers[camera_id]
    def process(self,camera_id:str,packet:Any):
        frame=packet.frame
        detections=self.last_detections.get(camera_id,[]);tracks=self.last_tracks.get(camera_id,[])
        # General YOLO is the expensive part of the live loop. Reuse the latest result
        # between inference frames while the camera feed itself continues at full rate.
        if packet.frame_index%self.detection_interval==0:
            try:
                with self._inference_lock:detections=self.detector.detect(frame);tracks=self._tracker_for(camera_id).update(detections,frame)
            except Exception:logger.exception('Analytics inference failed for %s',camera_id)
        self.last_detections[camera_id]=detections;self.last_tracks[camera_id]=tracks
        if packet.frame_index%self.anpr_scan_interval==0:
            try:self.last_anpr[camera_id]=self.anpr.scan(frame,detections,camera_id=camera_id,timestamp=packet.timestamp)
            except Exception:logger.exception('ANPR failed for %s',camera_id);self.last_anpr[camera_id]=[]
        if packet.frame_index%self.face_scan_interval==0:
            try:self.last_faces[camera_id]=self.face_service.detect(frame)
            except Exception:self.last_faces[camera_id]=[]
        if self.drone_enabled and packet.frame_index%self.drone_scan_interval==0:
            try:
                aerial=[d for d in detections if d.label.lower() in AERIAL_LABELS]
                with self._inference_lock:found=self.drone_detector.detect(frame,aerial_candidates=aerial)
                combined=[Detection(label='aerial_object',confidence=d.confidence,bbox=d.bbox) for d in aerial]+found
                self.last_drone_detections[camera_id]=self._dedupe_aerial(combined)
            except Exception:logger.exception('Aerial inference failed for %s',camera_id)
        aerial_now=self.last_drone_detections.get(camera_id,[])
        annotated=frame.copy()
        for d in detections:
            if d.label.lower() in AERIAL_LABELS:continue
            x1,y1,x2,y2=map(int,d.bbox);cv2.rectangle(annotated,(x1,y1),(x2,y2),(255,255,255),2);cv2.putText(annotated,f'{d.label} {d.confidence:.0%}',(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
        # Never expose the generic track number as the aerial classification. Aerial
        # detections get their own operator-facing label and box.
        for t in tracks:
            if t.label.lower() in AERIAL_LABELS:continue
            x1,y1,x2,y2=map(int,t.bbox);cv2.putText(annotated,f'#{t.track_id}',(x1,min(annotated.shape[0]-5,y2+18)),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA)
        for x,y,w,h in self.last_faces.get(camera_id,[]):cv2.rectangle(annotated,(x,y),(x+w,y+h),(255,180,0),2);cv2.putText(annotated,'FACE',(x,max(16,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,180,0),1,cv2.LINE_AA)
        for p in self.last_anpr.get(camera_id,[]):
            bx=p.get('plate_bbox') or p.get('bbox');x1,y1,x2,y2=map(int,bx);cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,255,255),2);label=f"PLATE {p['plate_text']}" if p.get('plate_text') else 'PLATE CANDIDATE';cv2.putText(annotated,label,(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2,cv2.LINE_AA)
        for d in aerial_now:
            x1,y1,x2,y2=map(int,d.bbox);cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,0,255),2);cv2.putText(annotated,f'AERIAL OBJECT {d.confidence:.0%}',(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2,cv2.LINE_AA)
        ok,encoded=cv2.imencode('.jpg',annotated,[cv2.IMWRITE_JPEG_QUALITY,76])
        if ok:self.frame_store.put(camera_id,FrameSnapshot(jpeg=encoded.tobytes(),frame_index=packet.frame_index,timestamp=packet.timestamp,detections=len(detections),tracks=len(tracks)))
        return detections,tracks
    @staticmethod
    def _dedupe_aerial(items):
        out=[]
        def iou(a,b):
            ax1,ay1,ax2,ay2=a;bx1,by1,bx2,by2=b;ix1,iy1,ix2,iy2=max(ax1,bx1),max(ay1,by1),min(ax2,bx2),min(ay2,by2);inter=max(0,ix2-ix1)*max(0,iy2-iy1)
            if not inter:return 0.0
            aa=max(0,ax2-ax1)*max(0,ay2-ay1);ab=max(0,bx2-bx1)*max(0,by2-by1);return inter/max(aa+ab-inter,1e-6)
        for d in sorted(items,key=lambda x:x.confidence,reverse=True):
            if all(iou(d.bbox,e.bbox)<.45 for e in out):out.append(d)
        return out
