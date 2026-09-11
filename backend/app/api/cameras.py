from fastapi import APIRouter, HTTPException, status

from app.schemas.camera import Camera, CameraCreate, CameraUpdate
from app.services.runtime import camera_manager, video_processor

router = APIRouter(prefix="/cameras", tags=["cameras"])


@router.get("", response_model=list[Camera])
def list_cameras() -> list[Camera]:
    return camera_manager.list()


@router.post("", response_model=Camera, status_code=status.HTTP_201_CREATED)
def create_camera(data: CameraCreate) -> Camera:
    try:
        camera = camera_manager.create(data)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    # Adding a source is an explicit operator action, so begin processing
    # immediately. The processor publishes the first frame before expensive AI
    # inference, making upload/local video sources feel instant in the UI.
    if camera.enabled:
        video_processor.start(camera.camera_id)
    return camera_manager.get(camera.camera_id)


@router.get("/{camera_id}", response_model=Camera)
def get_camera(camera_id: str) -> Camera:
    camera = camera_manager.get(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.patch("/{camera_id}", response_model=Camera)
def update_camera(camera_id: str, data: CameraUpdate) -> Camera:
    camera = camera_manager.update(camera_id, data)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    return camera


@router.delete("/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_camera(camera_id: str) -> None:
    if video_processor.is_running(camera_id):
        video_processor.stop(camera_id)
    if not camera_manager.delete(camera_id):
        raise HTTPException(status_code=404, detail="Camera not found")


@router.post("/{camera_id}/start", response_model=Camera)
def start_camera(camera_id: str) -> Camera:
    camera = camera_manager.get(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    if not video_processor.start(camera_id):
        raise HTTPException(status_code=409, detail="Camera is disabled or cannot be started")
    return camera_manager.get(camera_id)


@router.post("/{camera_id}/stop", response_model=Camera)
def stop_camera(camera_id: str) -> Camera:
    camera = camera_manager.get(camera_id)
    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")
    video_processor.stop(camera_id)
    return camera_manager.get(camera_id)
