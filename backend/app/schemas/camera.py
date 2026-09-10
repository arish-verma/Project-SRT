from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class CameraType(str, Enum):
    LOCAL = "LOCAL"
    WEBCAM = "WEBCAM"
    RTSP = "RTSP"
    HTTP = "HTTP"

class CameraStatus(str, Enum):
    OFFLINE = "OFFLINE"
    CONNECTING = "CONNECTING"
    ONLINE = "ONLINE"
    ERROR = "ERROR"

class CameraCreate(BaseModel):
    name: str = Field(min_length=1, max_length=100)
    source: str = Field(min_length=1, max_length=2048)
    location: str = Field(default="", max_length=200)
    enabled: bool = True

class CameraUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=100)
    source: str | None = Field(default=None, min_length=1, max_length=2048)
    location: str | None = Field(default=None, max_length=200)
    enabled: bool | None = None

class Camera(BaseModel):
    camera_id: str
    name: str
    source: str
    source_type: CameraType
    location: str = ""
    enabled: bool = True
    status: CameraStatus = CameraStatus.OFFLINE
    fps: float = 0.0
    frames_processed: int = 0
    last_frame_at: datetime | None = None
    error: str | None = None
