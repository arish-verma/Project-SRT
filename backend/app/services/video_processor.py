from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from app.ai.video import OpenCVVideoSource
from app.schemas.camera import CameraStatus
from app.services.camera_manager import CameraManager

logger = logging.getLogger(__name__)

@dataclass
class ProcessorState:
    thread: threading.Thread
    stop_event: threading.Event

class VideoProcessor:
    """Owns camera workers and provides a model-agnostic frame-processing hook."""
    def __init__(self, camera_manager: CameraManager) -> None:
        self.camera_manager = camera_manager
        self._workers: dict[str, ProcessorState] = {}
        self._lock = threading.RLock()

    def start(self, camera_id: str) -> bool:
        camera = self.camera_manager.get(camera_id)
        if not camera or not camera.enabled:
            return False
        with self._lock:
            state = self._workers.get(camera_id)
            if state and state.thread.is_alive():
                return True
            stop_event = threading.Event()
            thread = threading.Thread(target=self._run, args=(camera_id, stop_event),
                                      name=f"srt-camera-{camera_id}", daemon=True)
            self._workers[camera_id] = ProcessorState(thread, stop_event)
            thread.start()
            return True

    def stop(self, camera_id: str) -> bool:
        with self._lock:
            state = self._workers.get(camera_id)
            if not state:
                return False
            state.stop_event.set()
            return True

    def stop_all(self) -> None:
        with self._lock:
            for state in self._workers.values():
                state.stop_event.set()

    def is_running(self, camera_id: str) -> bool:
        with self._lock:
            state = self._workers.get(camera_id)
            return bool(state and state.thread.is_alive())

    def _run(self, camera_id: str, stop_event: threading.Event) -> None:
        camera = self.camera_manager.get(camera_id)
        if not camera:
            return
        self.camera_manager.set_runtime(camera_id, status=CameraStatus.CONNECTING, error=None)
        source = OpenCVVideoSource(camera.source, source_id=camera_id)
        started = time.monotonic()
        frames = 0
        failed = False
        try:
            for packet in source.frames():
                if stop_event.is_set():
                    break
                frames += 1
                elapsed = max(time.monotonic() - started, 0.001)
                self.camera_manager.set_runtime(
                    camera_id, status=CameraStatus.ONLINE,
                    fps=frames / elapsed, frames_processed=frames,
                )
                self.process_frame(camera_id, packet)
        except Exception as exc:
            failed = True
            logger.exception("Camera %s processing failed", camera_id)
            self.camera_manager.set_runtime(camera_id, status=CameraStatus.ERROR,
                                            frames_processed=frames, error=str(exc))
        finally:
            source.close()
            with self._lock:
                self._workers.pop(camera_id, None)
            if not self.camera_manager.get(camera_id):
                return
            if not failed:
                self.camera_manager.set_runtime(camera_id, status=CameraStatus.OFFLINE,
                                                frames_processed=frames, error=None)

    def process_frame(self, camera_id: str, packet) -> None:
        """Extension point for Phase 2 detection/tracking. Intentionally no model here."""
        return None
