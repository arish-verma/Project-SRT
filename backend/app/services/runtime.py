from __future__ import annotations

from app.core.config import settings
from app.services.alert_store import AlertStore
from app.services.camera_manager import CameraManager
from app.services.event_engine import RuleEventEngine
from app.services.event_store import EventStore
from app.services.frame_store import FrameStore
from app.services.zone_manager import ZoneManager
from app.services.video_processor import VideoProcessor

camera_manager = CameraManager()
zone_manager = ZoneManager()
event_store = EventStore()
alert_store = AlertStore()
frame_store = FrameStore()
event_engine = RuleEventEngine()
video_processor = VideoProcessor(camera_manager, frame_store, zone_manager, event_engine, event_store, settings.model_path)
