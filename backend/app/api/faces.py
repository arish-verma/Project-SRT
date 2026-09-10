from fastapi import APIRouter, File, UploadFile
import cv2
import numpy as np

from app.services.face_service import FaceDetectionService

router = APIRouter(prefix="/faces", tags=["faces"])
face_service = FaceDetectionService()

@router.post("/detect")
async def detect_faces(file: UploadFile = File(...)):
    data = await file.read()
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        return {"faces": [], "error": "Invalid image"}
    boxes = face_service.detect(frame)
    return {"faces": [{"x": x, "y": y, "width": w, "height": h} for x, y, w, h in boxes], "count": len(boxes)}
