from __future__ import annotations

import json
import sqlite3
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import RLock

from app.schemas.event import EventRecord, EventSeverity


class EventStore:
    """Thread-safe event repository with a bounded memory cache and SQLite persistence.

    SQLite keeps the hackathon deployment self-contained while preserving a clean
    repository boundary that can later be backed by PostgreSQL.
    """

    def __init__(self, max_events: int = 2000, db_path: str = "storage/srt_events.db") -> None:
        self._events: deque[EventRecord] = deque(maxlen=max_events)
        self._lock = RLock()
        self._db_path = Path(db_path)
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self._load_cache(max_events)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS events (
                    event_id TEXT PRIMARY KEY,
                    camera_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    risk_score INTEGER NOT NULL,
                    object_type TEXT NOT NULL,
                    track_id INTEGER,
                    zone_id TEXT,
                    zone_name TEXT,
                    message TEXT NOT NULL,
                    evidence_frame TEXT,
                    metadata_json TEXT NOT NULL
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_camera ON events(camera_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type)")

    @staticmethod
    def _from_row(row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            event_id=row["event_id"],
            camera_id=row["camera_id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            event_type=row["event_type"],
            severity=EventSeverity(row["severity"]),
            confidence=row["confidence"],
            risk_score=row["risk_score"],
            object_type=row["object_type"],
            track_id=row["track_id"],
            zone_id=row["zone_id"],
            zone_name=row["zone_name"],
            message=row["message"],
            evidence_frame=row["evidence_frame"],
            metadata=json.loads(row["metadata_json"] or "{}"),
        )

    def _load_cache(self, max_events: int) -> None:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM events ORDER BY timestamp DESC LIMIT ?", (max_events,)
            ).fetchall()
        for row in reversed(rows):
            self._events.appendleft(self._from_row(row))

    def add(self, event: EventRecord) -> None:
        with self._lock:
            self._events.appendleft(event)
            with self._connect() as conn:
                conn.execute(
                    """INSERT OR REPLACE INTO events
                    (event_id,camera_id,timestamp,event_type,severity,confidence,risk_score,
                     object_type,track_id,zone_id,zone_name,message,evidence_frame,metadata_json)
                    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        event.event_id,
                        event.camera_id,
                        event.timestamp.isoformat(),
                        event.event_type,
                        event.severity.value,
                        event.confidence,
                        event.risk_score,
                        event.object_type,
                        event.track_id,
                        event.zone_id,
                        event.zone_name,
                        event.message,
                        event.evidence_frame,
                        json.dumps(event.metadata, default=str),
                    ),
                )

    def list(
        self,
        camera_id: str | None = None,
        event_type: str | None = None,
        limit: int = 100,
    ) -> list[EventRecord]:
        safe_limit = max(1, min(limit, 500))
        with self._lock:
            with self._connect() as conn:
                clauses: list[str] = []
                params: list[str | int] = []
                if camera_id:
                    clauses.append("camera_id = ?")
                    params.append(camera_id)
                if event_type:
                    clauses.append("event_type = ?")
                    params.append(event_type)
                where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
                rows = conn.execute(
                    f"SELECT * FROM events {where} ORDER BY timestamp DESC LIMIT ?",  # noqa: S608
                    (*params, safe_limit),
                ).fetchall()
                return [self._from_row(row) for row in rows]

    def count(self) -> int:
        with self._lock:
            with self._connect() as conn:
                return int(conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
