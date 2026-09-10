from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.services.runtime import camera_manager, frame_store

router = APIRouter(prefix="/cameras", tags=["streams"])


def _mjpeg(camera_id: str):
    last = -1
    while True:
        snapshot = frame_store.wait_for(camera_id, after_frame=last, timeout=2.0)
        if snapshot is None:
            if not camera_manager.get(camera_id):
                return
            time.sleep(0.05)
            continue
        last = snapshot.frame_index
        yield (b"--frame\r\nContent-Type: image/jpeg\r\nContent-Length: " +
               str(len(snapshot.jpeg)).encode() + b"\r\n\r\n" + snapshot.jpeg + b"\r\n")


@router.get("/{camera_id}/stream")
def camera_stream(camera_id: str):
    if not camera_manager.get(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")
    return StreamingResponse(_mjpeg(camera_id), media_type="multipart/x-mixed-replace; boundary=frame")
