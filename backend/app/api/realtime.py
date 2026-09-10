from __future__ import annotations

import asyncio
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.services.runtime import event_store, alert_store

router = APIRouter(tags=["realtime"])

@router.websocket("/ws/events")
async def event_stream(websocket: WebSocket):
    await websocket.accept()
    seen: set[str] = set()
    try:
        while True:
            events = event_store.list(limit=50)
            for event in reversed(events):
                if event.event_id not in seen:
                    await websocket.send_json(event.model_dump(mode="json"))
                    seen.add(event.event_id)
            if len(seen) > 500:
                seen = set(list(seen)[-250:])
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        return

@router.websocket("/ws/alerts")
async def alert_stream(websocket: WebSocket):
    await websocket.accept()
    seen: set[str] = set()
    try:
        while True:
            alerts = alert_store.list(limit=50)
            for alert in reversed(alerts):
                if alert.alert_id not in seen:
                    await websocket.send_json(alert.model_dump(mode="json"))
                    seen.add(alert.alert_id)
            await asyncio.sleep(0.5)
    except (WebSocketDisconnect, RuntimeError):
        return
