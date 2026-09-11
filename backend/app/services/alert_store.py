from __future__ import annotations
import sqlite3
from datetime import datetime,timezone
from pathlib import Path
from threading import RLock
from uuid import uuid4
from app.schemas.alert import AlertRecord,AlertStatus
from app.schemas.event import EventRecord
class AlertStore:
 def __init__(self,db_path="storage/srt_events.db"):
  self.path=Path(db_path);self.path.parent.mkdir(parents=True,exist_ok=True);self.lock=RLock()
  with sqlite3.connect(self.path) as db:
   db.execute("CREATE TABLE IF NOT EXISTS alerts (alert_id TEXT PRIMARY KEY,event_id TEXT NOT NULL UNIQUE,camera_id TEXT NOT NULL,created_at TEXT NOT NULL,severity TEXT NOT NULL,title TEXT NOT NULL,message TEXT NOT NULL,status TEXT NOT NULL)");db.execute("CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at DESC)")
 def ensure_for_event(self,event:EventRecord):
  with self.lock,sqlite3.connect(self.path) as db:
   row=db.execute("SELECT * FROM alerts WHERE event_id=?",(event.event_id,)).fetchone()
   if row:return self._row(row)
   titles={"INTRUSION":"Restricted Border Entered","NIGHT_MOVEMENT":"Night Movement Detected","LOITERING":"Loitering Detected","MULTI_PERSON_ACTIVITY":"Multiple-Person Activity Detected","ANOMALY_SUSPECTED":"Anomaly Detected","FIGHT_SUSPECTED":"Potential Physical Altercation","AERIAL_OBJECT_DETECTED":"Aerial Object Detected","DRONE_DETECTED":"Aerial Object Detected","ANPR_DETECTED":"ANPR Vehicle Detected"}
   a=AlertRecord(alert_id=f"ALT-{uuid4().hex[:10].upper()}",event_id=event.event_id,camera_id=event.camera_id,created_at=datetime.now(timezone.utc),severity=event.severity,title=titles.get(event.event_type,f"{event.event_type.replace('_',' ').title()} Detected"),message=event.message)
   db.execute("INSERT INTO alerts VALUES (?,?,?,?,?,?,?,?)",(a.alert_id,a.event_id,a.camera_id,a.created_at.isoformat(),a.severity.value,a.title,a.message,a.status.value));return a
 @staticmethod
 def _row(r):return AlertRecord(alert_id=r[0],event_id=r[1],camera_id=r[2],created_at=datetime.fromisoformat(r[3]),severity=r[4],title=r[5],message=r[6],status=r[7])
 def list(self,status=None,limit=100):
  with self.lock,sqlite3.connect(self.path) as db:
   sql="SELECT * FROM alerts";args=[]
   if status:sql+=" WHERE status=?";args.append(status.value)
   sql+=" ORDER BY created_at DESC LIMIT ?";args.append(max(1,min(limit,500)));return [self._row(r) for r in db.execute(sql,args).fetchall()]
 def update(self,alert_id,status):
  with self.lock,sqlite3.connect(self.path) as db:
   db.execute("UPDATE alerts SET status=? WHERE alert_id=?",(status.value,alert_id));r=db.execute("SELECT * FROM alerts WHERE alert_id=?",(alert_id,)).fetchone();return self._row(r) if r else None
 def delete(self,alert_id):
  with self.lock,sqlite3.connect(self.path) as db:
   if not db.execute("SELECT 1 FROM alerts WHERE alert_id=?",(alert_id,)).fetchone():return False
   db.execute("DELETE FROM alerts WHERE alert_id=?",(alert_id,));return True
 def clear(self):
  with self.lock,sqlite3.connect(self.path) as db:db.execute("DELETE FROM alerts")
