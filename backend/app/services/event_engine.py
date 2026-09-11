from __future__ import annotations
import math,time,uuid
from collections import defaultdict,deque
from datetime import datetime,timezone
import cv2
from app.ai.interfaces import Detection,Track
from app.schemas.event import EventRecord,EventSeverity
from app.schemas.zone import Zone
class RuleEventEngine:
 def __init__(self,cooldown_seconds:float=8.0):
  self.cooldown_seconds=cooldown_seconds;self._inside=set();self._last_event={};self._entered_at={};self._history=defaultdict(lambda:deque(maxlen=24));self._drone_tracks=defaultdict(dict);self._next_drone_id=defaultdict(lambda:1)
 @staticmethod
 def _contains_point(zone,x,y,width,height):
  return cv2.pointPolygonTest(__import__('numpy').array([(px*width,py*height) for px,py in zone.polygon],dtype='float32'),(float(x),float(y)),False)>=0
 @classmethod
 def _contains(cls,zone,track,width,height): return cls._contains_point(zone,*track.center,width,height)
 @staticmethod
 def _is_night(timestamp):
  h=datetime.fromtimestamp(timestamp).hour;return h>=22 or h<5
 @staticmethod
 def _direction(history):
  if len(history)<3:return None
  _,x0,y0=history[0];_,x1,y1=history[-1];dx,dy=x1-x0,y1-y0
  if math.hypot(dx,dy)<8:return 'STATIONARY'
  h='EAST' if dx>0 else 'WEST';v='SOUTH' if dy>0 else 'NORTH'
  if abs(dx)<abs(dy)*.5:return v
  if abs(dy)<abs(dx)*.5:return h
  return f'{v}-{h}'
 @staticmethod
 def _movement_distance(history):
  return 0.0 if len(history)<2 else sum(math.hypot(history[i][1]-history[i-1][1],history[i][2]-history[i-1][2]) for i in range(1,len(history)))
 def _new_event(self,camera_id,now,event_type,severity,confidence,risk,message,object_type='person',track_id=None,zone=None,metadata=None):
  return EventRecord(event_id=f'EVT-{uuid.uuid4().hex[:10].upper()}',camera_id=camera_id,timestamp=datetime.fromtimestamp(now,tz=timezone.utc),event_type=event_type,severity=severity,confidence=confidence,risk_score=min(max(risk,0),100),object_type=object_type,track_id=track_id,zone_id=zone.zone_id if zone else None,zone_name=zone.name if zone else None,message=message,metadata=metadata or {})
 def _allow(self,key,now,cooldown=None):
  if now-self._last_event.get(key,0)<(cooldown if cooldown is not None else self.cooldown_seconds):return False
  self._last_event[key]=now;return True
 def evaluate(self,camera_id,tracks,zones,frame_shape,timestamp=None):
  now=timestamp or time.time();height,width=frame_shape[:2];events=[];current=set();night=self._is_night(now)
  for t in tracks:self._history[(camera_id,t.track_id)].append((now,t.center[0],t.center[1]))
  for zone in zones:
   if not zone.enabled or zone.camera_id!=camera_id:continue
   persons_inside=0
   for track in tracks:
    if track.label not in {'person','car','truck','bus','motorcycle'}:continue
    key=(camera_id,track.track_id,zone.zone_id);inside=self._contains(zone,track,width,height)
    if inside:current.add(key);persons_inside+=track.label=='person';self._entered_at.setdefault(key,now)
    if not inside:self._entered_at.pop(key,None);continue
    history=self._history[(camera_id,track.track_id)];direction=self._direction(history);dwell=max(0,now-self._entered_at.get(key,now));entry=key not in self._inside;movement=self._movement_distance(history)
    if entry and self._allow(key,now):
     risk=82 if zone.zone_type.value=='RESTRICTED' and track.label=='person' else 65;factors=['restricted zone entry'] if zone.zone_type.value=='RESTRICTED' else ['monitored zone entry']
     if night and track.label=='person':risk=min(100,risk+10);factors.append('night-time')
     if direction and direction!='STATIONARY':factors.append(f'direction {direction.lower()}')
     sev=EventSeverity.CRITICAL if risk>=90 else EventSeverity.HIGH if risk>=75 else EventSeverity.MEDIUM
     events.append(self._new_event(camera_id,now,'INTRUSION',sev,track.confidence,risk,f'{track.label.title()} entered {zone.name}',track.label,track.track_id,zone,{'zone_type':zone.zone_type.value,'factors':factors,'night':night,'dwell_seconds':round(dwell,1),'direction':direction}))
    if track.label=='person' and zone.zone_type.value=='RESTRICTED':
     if dwell>=30 and self._allow((camera_id,track.track_id,zone.zone_id,'loiter'),now,15) and movement<max(width,height)*.18:
      events.append(self._new_event(camera_id,now,'LOITERING',EventSeverity.HIGH,track.confidence,80,f'Person remained in {zone.name} for {int(dwell)}s with limited movement','person',track.track_id,zone,{'dwell_seconds':round(dwell,1),'movement_pixels':round(movement,1),'factors':['restricted zone','extended dwell','limited movement']}))
     if night and self._allow((camera_id,track.track_id,zone.zone_id,'night'),now,20):
      events.append(self._new_event(camera_id,now,'NIGHT_MOVEMENT',EventSeverity.HIGH,track.confidence,75,f'Night-time movement detected in {zone.name}','person',track.track_id,zone,{'factors':['night-time','restricted zone'],'direction':direction}))
     if len(history)>=8 and movement>max(width,height)*.75 and self._allow((camera_id,track.track_id,zone.zone_id,'anomaly'),now,20):
      events.append(self._new_event(camera_id,now,'ANOMALY_SUSPECTED',EventSeverity.HIGH,track.confidence,78,f'Unusual movement pattern detected in {zone.name}','person',track.track_id,zone,{'factors':['unusual movement speed','restricted zone'],'movement_pixels':round(movement,1),'direction':direction,'note':'Anomaly score is an operator-assistance signal, not proof of intent.'}))
   if persons_inside>=3 and self._allow((camera_id,zone.zone_id,'group'),now,20):
    events.append(self._new_event(camera_id,now,'MULTI_PERSON_ACTIVITY',EventSeverity.HIGH,.9,78,f'Multiple people detected in {zone.name}','person',None,zone,{'person_count':persons_inside,'factors':['restricted zone','multiple people']}))
  self._inside=current
  active={(camera_id,t.track_id) for t in tracks}
  for key in list(self._history):
   if key[0]==camera_id and key not in active and self._history[key] and now-self._history[key][-1][0]>10:del self._history[key]
  return events
 @staticmethod
 def _iou(a,b):
  ax1,ay1,ax2,ay2=a;bx1,by1,bx2,by2=b;ix1,iy1,ix2,iy2=max(ax1,bx1),max(ay1,by1),min(ax2,bx2),min(ay2,by2);inter=max(0,ix2-ix1)*max(0,iy2-iy1)
  if inter<=0:return 0.0
  aa=max(0,ax2-ax1)*max(0,ay2-ay1);ab=max(0,bx2-bx1)*max(0,by2-by1);return inter/max(aa+ab-inter,1e-6)
 def _assign_drone_tracks(self,camera_id,detections):
  previous=self._drone_tracks[camera_id];result=[];used=set()
  for d in detections:
   best_id,best_iou=None,.20
   for tid,bbox in previous.items():
    if tid in used:continue
    score=self._iou(bbox,d.bbox)
    if score>best_iou:best_id,best_iou=tid,score
   if best_id is None:best_id=self._next_drone_id[camera_id];self._next_drone_id[camera_id]+=1
   used.add(best_id);result.append((best_id,d))
  self._drone_tracks[camera_id]={tid:d.bbox for tid,d in result};return result
 def evaluate_drones(self,camera_id,detections,zones,frame_shape,timestamp=None):
  now=timestamp or time.time();height,width=frame_shape[:2];events=[];rz=next((z for z in zones if z.enabled and z.camera_id==camera_id and z.zone_type.value=='RESTRICTED'),None)
  for tid,d in self._assign_drone_tracks(camera_id,detections):
   x1,y1,x2,y2=d.bbox;cx,cy=(x1+x2)/2,(y1+y2)/2;inside=bool(rz and self._contains_point(rz,cx,cy,width,height))
   if not self._allow((camera_id,'drone',tid,'presence'),now,max(self.cooldown_seconds,10)):continue
   risk=90 if inside else 60;sev=EventSeverity.CRITICAL if inside else EventSeverity.MEDIUM
   events.append(self._new_event(camera_id,now,'DRONE_DETECTED',sev,d.confidence,risk,'Drone detected in restricted zone' if inside else 'Drone detected','drone',None,rz if inside else None,{'restricted_zone':inside,'drone_track_id':tid,'bbox':list(d.bbox),'factors':['drone detection']+(['restricted zone'] if inside else [])}))
  return events
