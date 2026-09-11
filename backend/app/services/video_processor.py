from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from app.ai.video import OpenCVVideoSource
from app.core.config import settings
from app.schemas.camera import CameraStatus
from app.services.analytics_pipeline import AnalyticsPipeline
from app.services.camera_manager import CameraManager
from app.services.event_engine import RuleEventEngine
from app.services.event_store import EventStore
from app.services.frame_store import FrameStore
from app.services.zone_manager import ZoneManager

logger = logging.getLogger(__name__)


@dataclass
class ProcessorState:
    thread: threading.Thread
    stop_event: threading.Event


class VideoProcessor:
    """Owns camera workers and connects ingestion to perception and event reasoning."""

    def __init__(self, camera_manager: CameraManager, frame_store: FrameStore | None = None, zone_manager: ZoneManager | None = None, event_engine: RuleEventEngine | None = None, event_store: EventStore | None = None, model_path: str | None = None) -> None:
        self.camera_manager = camera_manager
        self.frame_store = frame_store or FrameStore()
        self.zone_manager = zone_manager or ZoneManager()
        self.event_engine = event_engine or RuleEventEngine()
        self.event_store = event_store or EventStore()
        self.pipeline = AnalyticsPipeline(self.frame_store, model_path or settings.model_path, settings.model_confidence)
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
            thread = threading.Thread(target=self._run, args=(camera_id, stop_event), name=f"srt-camera-{camera_id}", daemon=True)
            self._workers[camera_id] = ProcessorState(thread, stop_event)
            thread.start()
            return True

    def start_all(self, camera_ids: list[str] | None = None) -> int:
        """Start all configured cameras concurrently so a large channel set can be armed with one action."""
        ids = camera_ids if camera_ids is not None else [c.camera_id for c in self.camera_manager.list() if c.enabled]
        if not ids:
            return 0
        with ThreadPoolExecutor(max_workers=min(32, len(ids)), thread_name_prefix="srt-start-all") as pool:
            results = list(pool.map(self.start, ids))
        return sum(1 for result in results if result)

    def stop(self, camera_id: str) -> bool:
        with self._lock:
            state = self._workers.get(camera_id)
            if not state:
                return False
            state.stop_event.set()
            return True

    def stop_all(self) -> int:
        with self._lock:
            states = list(self._workers.values())
            for state in states:
                state.stop_event.set()
            return len(states)

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
        started, frames, failed = time.monotonic(), 0, False
        try:
            for packet in source.frames():
                if stop_event.is_set():
                    break
                frames += 1
                elapsed = max(time.monotonic() - started, 0.001)
                self.camera_manager.set_runtime(camera_id, status=CameraStatus.ONLINE, fps=frames / elapsed, frames_processed=frames)
                self.process_frame(camera_id, packet)
        except Exception as exc:
            failed = True
            logger.exception("Camera %s processing failed", camera_id)
            self.camera_manager.set_runtime(camera_id, status=CameraStatus.ERROR, frames_processed=frames, error=str(exc))
        finally:
            source.close()
            with self._lock:
                self._workers.pop(camera_id, None)
            if self.camera_manager.get(camera_id) and not failed:
                self.camera_manager.set_runtime(camera_id, status=CameraStatus.OFFLINE, frames_processed=frames, error=None)

    def process_frame(self, camera_id: str, packet) -> None:
        _, tracks = self.pipeline.process(camera_id, packet)
        zones = self.zone_manager.list(camera_id)
        events = self.event_engine.evaluate(camera_id, tracks, zones, packet.frame.shape, packet.timestamp)
        events.extend(self.event_engine.evaluate_drones(camera_id, self.pipeline.last_drone_detections, zones, packet.frame.shape, packet.timestamp))
        snapshot = self.frame_store.get(camera_id)
        root = Path(settings.event_storage_path)
        for event in events:
            evidence_path = None
            if snapshot:
                root.mkdir(parents=True, exist_ok=True)
                evidence = root / f"{event.event_id}.jpg"
                evidence.write_bytes(snapshot.jpeg)
                evidence_path = str(evidence)
            event = event.model_copy(update={"evidence_frame": evidence_path})
            self.event_store.add(event)
            try:
                from app.services.runtime import alert_store
                alert_store.ensure_for_event(event)
            except Exception:
                logger.exception("Failed to persist alert for %s", event.event_id)
