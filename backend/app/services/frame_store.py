from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class FrameSnapshot:
    jpeg: bytes
    frame_index: int
    timestamp: float
    detections: int = 0
    tracks: int = 0


class FrameStore:
    """Thread-safe latest-frame cache used by the browser MJPEG stream."""

    def __init__(self) -> None:
        self._frames: dict[str, FrameSnapshot] = {}
        self._condition = threading.Condition()

    def put(self, camera_id: str, snapshot: FrameSnapshot) -> None:
        with self._condition:
            self._frames[camera_id] = snapshot
            self._condition.notify_all()

    def get(self, camera_id: str) -> FrameSnapshot | None:
        with self._condition:
            return self._frames.get(camera_id)

    def wait_for(self, camera_id: str, after_frame: int = -1, timeout: float = 2.0) -> FrameSnapshot | None:
        with self._condition:
            self._condition.wait_for(
                lambda: camera_id in self._frames and self._frames[camera_id].frame_index > after_frame,
                timeout=timeout,
            )
            return self._frames.get(camera_id)

    def remove(self, camera_id: str) -> None:
        with self._condition:
            self._frames.pop(camera_id, None)
