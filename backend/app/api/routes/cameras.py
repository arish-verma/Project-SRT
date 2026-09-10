from fastapi import APIRouter, HTTPException
from app.schemas.domain import Camera

router = APIRouter()
_cameras: dict[str, Camera] = {}

@router.get("", response_model=list[Camera])
def list_cameras() -> list[Camera]:
    return list(_cameras.values())

@router.post("", response_model=Camera, status_code=201)
def create_camera(camera: Camera) -> Camera:
    if camera.camera_id in _cameras:
        raise HTTPException(status_code=409, detail="camera_id already exists")
    _cameras[camera.camera_id] = camera
    return camera

@router.get("/{camera_id}", response_model=Camera)
def get_camera(camera_id: str) -> Camera:
    camera = _cameras.get(camera_id)
    if camera is None:
        raise HTTPException(status_code=404, detail="camera not found")
    return camera
