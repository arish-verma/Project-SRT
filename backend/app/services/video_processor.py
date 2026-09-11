from __future__ import annotations

import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.ai.video import OpenCVVideoSource
from app.core.config import settings
from app.schemas.camera import CameraStatus
from app.services.analytics_pipeline import AnalyticsPipeline
from app.services.camera_manager import CameraManager
from app.services.event_engine import RuleEventEngine
from app.services.event_store import EventStore
from app.services.frame_store import FrameSnapshot, FrameStore
from app.services.zone_manager import ZoneManager

logger = logging.getLogger(__name__)


@dataclass
class ProcessorState:
    thread: threading.Thread
    stop_event: threading.Event


class VideoProcessor:
    def __init__(
        self,
        camera_manager,
        frame_store=None,
        zone_manager=None,
        event_engine=None,
        event_store=None,
        model_path=None,
    ):
        self.camera_manager = camera_manager
        self.frame_store = frame_store or FrameStore()
        self.zone_manager = zone_manager or ZoneManager()
        self.event_engine = event_engine or RuleEventEngine()
        self.event_store = event_store or EventStore()
        self.pipeline = AnalyticsPipeline(
            self.frame_store,
            model_path or settings.model_path,
            settings.model_confidence,
        )
        self._workers: dict[str, ProcessorState] = {}
        self._lock = threading.RLock()
        self._record_buffers = defaultdict(lambda: deque(maxlen=40))

    def start(self, camera_id):
        camera = self.camera_manager.get(camera_id)
        if not camera or not camera.enabled:
            return False
        with self._lock:
            state = self._workers.get(camera_id)
            if state and state.thread.is_alive():
                return True
            stop_event = threading.Event()
            state = ProcessorState(
                thread=threading.Thread(
                    target=self._run,
                    args=(camera_id, stop_event),
                    name=f"srt-camera-{camera_id}",
                    daemon=True,
                ),
                stop_event=stop_event,
            )
            self._workers[camera_id] = state
            state.thread.start()
            return True

    def start_all(self, camera_ids=None):
        ids = camera_ids if camera_ids is not None else [camera.camera_id for camera in self.camera_manager.list() if camera.enabled]
        if not ids:
            return 0
        with ThreadPoolExecutor(max_workers=min(32, len(ids)), thread_name_prefix="srt-start-all") as pool:
            return sum(pool.map(self.start, ids))

    def stop(self, camera_id):
        with self._lock:
            state = self._workers.get(camera_id)
            if not state:
                return False
            state.stop_event.set()
            return True

    def stop_all(self):
        with self._lock:
            for state in self._workers.values():
                state.stop_event.set()
            return len(self._workers)

    def is_running(self, camera_id):
        with self._lock:
            state = self._workers.get(camera_id)
            return bool(state and state.thread.is_alive())

    def _publish_raw_frame(self, camera_id, packet):
        ok, encoded = cv2.imencode(".jpg", packet.frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            self.frame_store.put(
                camera_id,
                FrameSnapshot(
                    jpeg=encoded.tobytes(),
                    frame_index=packet.frame_index,
                    timestamp=packet.timestamp,
                ),
            )

    def _run(self, camera_id, stop_event):
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
                    camera_id,
                    status=CameraStatus.ONLINE,
                    fps=frames / elapsed,
                    frames_processed=frames,
                )

                # Publish immediately so a heavy first model load can never make
                # the browser feed appear dead/blank.
                self._publish_raw_frame(camera_id, packet)

                try:
                    self.process_frame(camera_id, packet)
                except Exception:
                    # A bad analytics frame must not kill the camera transport.
                    # The raw feed remains available and the next frame is tried.
                    logger.exception("Analytics frame failed for %s at frame %s", camera_id, packet.frame_index)
        except Exception as exc:
            failed = True
            logger.exception("Camera %s processing failed", camera_id)
            self.camera_manager.set_runtime(
                camera_id,
                status=CameraStatus.ERROR,
                frames_processed=frames,
                error=str(exc),
            )
        finally:
            source.close()
            with self._lock:
                self._workers.pop(camera_id, None)
            if self.camera_manager.get(camera_id) and not failed:
                self.camera_manager.set_runtime(
                    camera_id,
                    status=CameraStatus.OFFLINE,
                    frames_processed=frames,
                    error=None,
                )

    def _write_clip(self, camera_id, event_id):
        buffer = list(self._record_buffers.get(camera_id, []))
        if len(buffer) < 3:
            return None
        root = Path(settings.event_storage_path)
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"{event_id}.mp4"
        first = cv2.imdecode(np.frombuffer(buffer[0][1], np.uint8), cv2.IMREAD_COLOR)
        if first is None:
            return None
        height, width = first.shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (width, height))
        try:
            for _, jpeg in buffer:
                frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    writer.write(frame)
        finally:
            writer.release()
        return str(path) if path.exists() else None

    def process_frame(self, camera_id, packet):
        _, tracks = self.pipeline.process(camera_id, packet)
        snapshot = self.frame_store.get(camera_id)
        if snapshot:
            self._record_buffers[camera_id].append((packet.timestamp, snapshot.jpeg))

        zones = self.zone_manager.list(camera_id)
        events = self.event_engine.evaluate(
            camera_id,
            tracks,
            zones,
            packet.frame.shape,
            packet.timestamp,
        )
        events.extend(
            self.event_engine.evaluate_drones(
                camera_id,
                self.pipeline.last_drone_detections.get(camera_id, []),
                zones,
                packet.frame.shape,
                packet.timestamp,
            )
        )

        root = Path(settings.event_storage_path)
        for event in events:
            evidence_path = None
            clip_path = None
            if snapshot:
                root.mkdir(parents=True, exist_ok=True)
                evidence = root / f"{event.event_id}.jpg"
                evidence.write_bytes(snapshot.jpeg)
                evidence_path = str(evidence)
                clip_path = self._write_clip(camera_id, event.event_id)
            metadata = dict(event.metadata)
            metadata["recording_clip"] = clip_path
            metadata["evidence_available"] = bool(evidence_path)
            event = event.model_copy(update={"evidence_frame": evidence_path, "metadata": metadata})
            self.event_store.add(event)
            try:
                from app.services.runtime import alert_store

                alert_store.ensure_for_event(event)
            except Exception:
                logger.exception("Failed to persist alert for %s", event.event_id)
