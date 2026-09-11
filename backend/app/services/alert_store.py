from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4

from app.schemas.alert import AlertRecord, AlertStatus
from app.schemas.event import EventRecord


class AlertStore:
    """Persistent alert repository backed by the same self-contained SQLite database."""
    def __init__(self, db_path: str = "storage/srt_events.db") -> None:
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = RLock()
        with sqlite3.connect(self.path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS alerts (
                alert_id TEXT PRIMARY KEY, event_id TEXT NOT NULL UNIQUE, camera_id TEXT NOT NULL,
                created_at TEXT NOT NULL, severity TEXT NOT NULL, title TEXT NOT NULL,
                message TEXT NOT NULL, status TEXT NOT NULL)""")
            db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC)")

    def ensure_for_event(self, event: EventRecord) -> AlertRecord:
        with self.lock, sqlite3.connect(self.path) as db:
            row = db.execute("SELECT * FROM alerts WHERE event_id=?", (event.event_id,)).fetchone()
            if row:
                return self._row(row)
            title_map = {
                "INTRUSION": "Restricted Border Entered",
                "NIGHT_MOVEMENT": "Night Movement Detected",
                "LOITERING": "Loitering Detected",
                "MULTI_PERSON_ACTIVITY": "Multiple-Person Activity Detected",
                "DRONE_DETECTED": "Drone Detected",
            }
            title = title_map.get(event.event_type, f"{event.event_type.replace('_', ' ').title()} Detected")
            alert = AlertRecord(
                alert_id=f"ALT-{uuid4().hex[:10].upper()}",
                event_id=event.event_id,
                camera_id=event.camera_id,
                created_at=datetime.now(timezone.utc),
                severity=event.severity,
                title=title,
                message=event.message,
            )
            db.execute("INSERT INTO alerts VALUES (?,?,?,?,?,?,?,?)", (alert.alert_id, alert.event_id,
                alert.camera_id, alert.created_at.isoformat(), alert.severity.value, alert.title,
                alert.message, alert.status.value))
            return alert

    @staticmethod
    def _row(row) -> AlertRecord:
        return AlertRecord(alert_id=row[0], event_id=row[1], camera_id=row[2], created_at=datetime.fromisoformat(row[3]),
            severity=row[4], title=row[5], message=row[6], status=row[7])

    def list(self, status: AlertStatus | None = None, limit: int = 100) -> list[AlertRecord]:
        with self.lock, sqlite3.connect(self.path) as db:
            sql = "SELECT * FROM alerts"
            args = []
            if status:
                sql += " WHERE status=?"; args.append(status.value)
            sql += " ORDER BY created_at DESC LIMIT ?"; args.append(max(1, min(limit, 500)))
            return [self._row(r) for r in db.execute(sql, args).fetchall()]

    def update(self, alert_id: str, status: AlertStatus) -> AlertRecord | None:
        with self.lock, sqlite3.connect(self.path) as db:
            db.execute("UPDATE alerts SET status=? WHERE alert_id=?", (status.value, alert_id))
            row = db.execute("SELECT * FROM alerts WHERE alert_id=?", (alert_id,)).fetchone()
            return self._row(row) if row else None
