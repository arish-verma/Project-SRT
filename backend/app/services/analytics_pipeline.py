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
class AnalyticsPipeline:
 def __init__(self,frame_store:FrameStore,model_path:str='yolo11n.pt',confidence:float=.35)->None:
  self.frame_store=frame_store;self.model_path=model_path;self.confidence=confidence;self.detector=YOLODetector(model_path,confidence);self._trackers={};self._tracker_lock=threading.RLock();self._inference_lock=threading.RLock();self.drone_detector=DroneDetector(settings.drone_model_path,settings.drone_model_confidence);self.drone_scan_interval=max(1,settings.drone_scan_interval);self.drone_enabled=False;self.last_drone_detections=[];self.last_detections={};self.last_tracks={};self.last_anpr={};self.last_faces={};self.anpr=ANPREngine();self.face_service=FaceDetectionService()
 def _tracker_for(self,camera_id:str):
  with self._tracker_lock:
   if camera_id not in self._trackers:self._trackers[camera_id]=ByteTrackTracker(self.model_path,self.confidence)
   return self._trackers[camera_id]
 def process(self,camera_id:str,packet:Any):
  frame=packet.frame;detections=[];tracks=[];self.last_drone_detections=[]
  try:
   with self._inference_lock:detections=self.detector.detect(frame);tracks=self._tracker_for(camera_id).update(detections,frame)
  except Exception:logger.exception('Analytics inference failed for %s',camera_id)
  self.last_detections[camera_id]=detections;self.last_tracks[camera_id]=tracks
  if packet.frame_index%15==0:
   try:self.last_anpr[camera_id]=self.anpr.scan(frame,detections)
   except Exception:self.last_anpr[camera_id]=[]
  if packet.frame_index%5==0:
   try:self.last_faces[camera_id]=self.face_service.detect(frame)
   except Exception:self.last_faces[camera_id]=[]
  if self.drone_enabled and packet.frame_index%self.drone_scan_interval==0:
   try:
    aerial=[d for d in detections if d.label.lower() in {'airplane','aircraft','helicopter'}]
    with self._inference_lock:self.last_drone_detections=self.drone_detector.detect(frame,aerial_candidates=aerial)
   except Exception:self.last_drone_detections=[]
  annotated=frame.copy()
  for d in detections:
   x1,y1,x2,y2=map(int,d.bbox);cv2.rectangle(annotated,(x1,y1),(x2,y2),(255,255,255),2);cv2.putText(annotated,f'{d.label} {d.confidence:.0%}',(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.5,(255,255,255),1,cv2.LINE_AA)
  for t in tracks:
   x1,y1,x2,y2=map(int,t.bbox);cv2.putText(annotated,f'#{t.track_id}',(x1,min(annotated.shape[0]-5,y2+18)),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2,cv2.LINE_AA)
  for x,y,w,h in self.last_faces.get(camera_id,[]):cv2.rectangle(annotated,(x,y),(x+w,y+h),(255,180,0),2);cv2.putText(annotated,'FACE',(x,max(16,y-5)),cv2.FONT_HERSHEY_SIMPLEX,.45,(255,180,0),1,cv2.LINE_AA)
  for p in self.last_anpr.get(camera_id,[]):
   if p.get('plate_text'):
    x1,y1,x2,y2=p['bbox'];cv2.putText(annotated,f"PLATE {p['plate_text']}",(x1,y2+36),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,255,255),2,cv2.LINE_AA)
  for d in self.last_drone_detections:
   x1,y1,x2,y2=map(int,d.bbox);cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,0,255),2);cv2.putText(annotated,f'DRONE {d.confidence:.0%}',(x1,max(18,y1-6)),cv2.FONT_HERSHEY_SIMPLEX,.55,(0,0,255),2,cv2.LINE_AA)
  ok,encoded=cv2.imencode('.jpg',annotated,[cv2.IMWRITE_JPEG_QUALITY,82])
  if ok:self.frame_store.put(camera_id,FrameSnapshot(jpeg=encoded.tobytes(),frame_index=packet.frame_index,timestamp=packet.timestamp,detections=len(detections),tracks=len(tracks)))
  return detections,tracks
