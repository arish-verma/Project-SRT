from __future__ import annotations

from collections import deque
from threading import RLock

from app.schemas.event import EventRecord


class EventStore:
    def __init__(self, max_events: int = 2000) -> None:
        self._events: deque[EventRecord] = deque(maxlen=max_events)
        self._lock = RLock()

    def add(self, event: EventRecord) -> None:
        with self._lock:
            self._events.appendleft(event)

    def list(self, camera_id: str | None = None, event_type: str | None = None, limit: int = 100) -> list[EventRecord]:
        with self._lock:
            result = [e for e in self._events if (camera_id is None or e.camera_id == camera_id) and (event_type is None or e.event_type == event_type)]
            return result[:max(1, min(limit, 500))]

    def count(self) -> int:
        with self._lock:
            return len(self._events)
