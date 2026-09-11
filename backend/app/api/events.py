from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from app.core.config import settings
from app.schemas.event import EventRecord
from app.services.runtime import event_store

router=APIRouter(prefix="/events",tags=["events"])

@router.get("",response_model=list[EventRecord])
def list_events(camera_id:str|None=None,event_type:str|None=None,limit:int=100)->list[EventRecord]: return event_store.list(camera_id=camera_id,event_type=event_type,limit=limit)

@router.get("/evidence/{event_id}")
def event_evidence(event_id:str):
    event=next((e for e in event_store.list(limit=500) if e.event_id==event_id),None)
    if not event or not event.evidence_frame: raise HTTPException(status_code=404,detail="Evidence not found")
    path=Path(event.evidence_frame)
    if not path.exists(): raise HTTPException(status_code=404,detail="Evidence file unavailable")
    return FileResponse(path,media_type="image/jpeg",filename=f"{event_id}.jpg")

@router.delete("/{event_id}",status_code=204)
def delete_event(event_id:str)->None:
    if not event_store.delete(event_id): raise HTTPException(status_code=404,detail="Event not found")
