from fastapi import APIRouter, HTTPException
from app.services.runtime import video_processor

router=APIRouter(prefix="/anpr",tags=["anpr"])

@router.get("/camera/{camera_id}")
def camera_anpr(camera_id:str):
    if not video_processor.camera_manager.get(camera_id): raise HTTPException(status_code=404,detail="Camera not found")
    return {"camera_id":camera_id,"enabled":True,"results":video_processor.pipeline.last_anpr.get(camera_id,[]),"note":"OCR results are emitted only when text is confidently readable; otherwise the system keeps a plate candidate."}

@router.get("/summary")
def anpr_summary():
    results=[]
    for camera_id,items in video_processor.pipeline.last_anpr.items():
        for item in items: results.append({"camera_id":camera_id,**item})
    return {"results":results,"cameras_scanned":len(video_processor.pipeline.last_anpr)}
