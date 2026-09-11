from fastapi import APIRouter
from app.services.runtime import camera_manager,event_store,video_processor
router=APIRouter(prefix='/analytics',tags=['analytics'])
@router.get('/summary')
def summary():
 cameras=camera_manager.list();online=sum(1 for c in cameras if c.status.value=='ONLINE');events=event_store.list(limit=500);faces=sum(len(v) for v in video_processor.pipeline.last_faces.values());tracks=sum(len(v) for v in video_processor.pipeline.last_tracks.values());vehicles=sum(1 for items in video_processor.pipeline.last_detections.values() for d in items if d.label in {'car','truck','bus','motorcycle'})
 return {'cameras_total':len(cameras),'cameras_online':online,'events_total':event_store.count(),'events_recent':len(events),'high_priority':sum(1 for e in events if e.severity.value in {'HIGH','CRITICAL'}),'intrusions':sum(1 for e in events if e.event_type=='INTRUSION'),'faces_detected':faces,'tracked_objects':tracks,'vehicles_detected':vehicles,'by_type':{k:sum(1 for e in events if e.event_type==k) for k in sorted({e.event_type for e in events})}}
