from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from threading import RLock
from uuid import uuid4

from app.schemas.camera import Camera, CameraCreate, CameraStatus, CameraType, CameraUpdate


def infer_source_type(source: str) -> CameraType:
    value = source.strip().lower()
    if value.isdigit():
        return CameraType.WEBCAM
    if value.startswith("rtsp://"):
        return CameraType.RTSP
    if value.startswith(("http://", "https://")):
        return CameraType.HTTP
    return CameraType.LOCAL


@dataclass
class CameraRecord:
    camera: Camera


class CameraManager:
    """Runtime camera registry. Persistence is intentionally deferred to Phase 2."""
    def __init__(self) -> None:
        self._records: dict[str, CameraRecord] = {}
        self._lock = RLock()

    def list(self) -> list[Camera]:
        with self._lock:
            return [r.camera.model_copy(deep=True) for r in self._records.values()]

    def get(self, camera_id: str) -> Camera | None:
        with self._lock:
            record = self._records.get(camera_id)
            return record.camera.model_copy(deep=True) if record else None

    def create(self, data: CameraCreate) -> Camera:
        camera = Camera(
            camera_id=f"CAM-{uuid4().hex[:8].upper()}",
            name=data.name,
            source=data.source,
            source_type=infer_source_type(data.source),
            location=data.location,
            enabled=data.enabled,
        )
        with self._lock:
            self._records[camera.camera_id] = CameraRecord(camera)
        return camera.model_copy(deep=True)

    def update(self, camera_id: str, data: CameraUpdate) -> Camera | None:
        with self._lock:
            record = self._records.get(camera_id)
            if not record:
                return None
            values = data.model_dump(exclude_unset=True)
            if "source" in values:
                values["source_type"] = infer_source_type(values["source"])
                values["status"] = CameraStatus.OFFLINE
            record.camera = record.camera.model_copy(update=values)
            return record.camera.model_copy(deep=True)

    def delete(self, camera_id: str) -> bool:
        with self._lock:
            return self._records.pop(camera_id, None) is not None

    def set_runtime(self, camera_id: str, *, status: CameraStatus, fps: float | None = None,
                    frames_processed: int | None = None, error: str | None = None) -> None:
        with self._lock:
            record = self._records.get(camera_id)
            if not record:
                return
            values = {"status": status, "error": error}
            if fps is not None:
                values["fps"] = max(0.0, fps)
            if frames_processed is not None:
                values["frames_processed"] = max(0, frames_processed)
            if status == CameraStatus.ONLINE:
                values["last_frame_at"] = datetime.now(timezone.utc)
            record.camera = record.camera.model_copy(update=values)
