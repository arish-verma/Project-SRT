from pathlib import Path
from fastapi import APIRouter,HTTPException
from fastapi.responses import FileResponse
from app.core.config import settings
from app.schemas.event import EventRecord
from app.services.runtime import event_store
router=APIRouter(prefix='/events',tags=['events'])
@router.get('',response_model=list[EventRecord])
def list_events(camera_id:str|None=None,event_type:str|None=None,limit:int=100): return event_store.list(camera_id=camera_id,event_type=event_type,limit=limit)
@router.get('/evidence/{event_id}')
def evidence(event_id:str):
 e=next((x for x in event_store.list(limit=500) if x.event_id==event_id),None)
 if not e or not e.evidence_frame:raise HTTPException(404,'Evidence not found')
 p=Path(e.evidence_frame)
 if not p.exists():raise HTTPException(404,'Evidence unavailable')
 return FileResponse(p,media_type='image/jpeg',filename=f'{event_id}.jpg')
@router.get('/recording/{event_id}')
def recording(event_id:str):
 e=next((x for x in event_store.list(limit=500) if x.event_id==event_id),None)
 p=Path((e.metadata or {}).get('recording_clip','')) if e else Path('')
 if not e or not p.exists():raise HTTPException(404,'Event recording unavailable')
 return FileResponse(p,media_type='video/mp4',filename=f'{event_id}.mp4')
@router.delete('/{event_id}',status_code=204)
def delete_event(event_id:str):
 if not event_store.delete(event_id):raise HTTPException(404,'Event not found')
