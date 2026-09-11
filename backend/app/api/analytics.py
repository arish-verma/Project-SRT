from fastapi import APIRouter

from app.services.runtime import camera_manager, event_store, video_processor

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/summary")
def summary():
    cameras = camera_manager.list()
    online = sum(1 for camera in cameras if camera.status.value == "ONLINE")
    events = event_store.list(limit=500)
    faces = sum(len(value) for value in video_processor.pipeline.last_faces.values())
    tracks = sum(len(value) for value in video_processor.pipeline.last_tracks.values())
    vehicles = sum(
        1
        for items in video_processor.pipeline.last_detections.values()
        for detection in items
        if detection.label in {"car", "truck", "bus", "motorcycle"}
    )
    flying = [
        {
            "camera_id": camera_id,
            "label": detection.label,
            "confidence": detection.confidence,
            "bbox": list(detection.bbox),
        }
        for camera_id, items in video_processor.pipeline.last_flying_detections.items()
        for detection in items
    ]
    return {
        "cameras_total": len(cameras),
        "cameras_online": online,
        "events_total": event_store.count(),
        "events_recent": len(events),
        "high_priority": sum(1 for event in events if event.severity.value in {"HIGH", "CRITICAL"}),
        "intrusions": sum(1 for event in events if event.event_type == "INTRUSION"),
        "drone_alerts": sum(1 for event in events if event.event_type == "DRONE_DETECTED"),
        "faces_detected": faces,
        "tracked_objects": tracks,
        "vehicles_detected": vehicles,
        "flying_objects": flying,
        "flying_model": video_processor.pipeline.drone_detector.status(),
        "by_type": {key: sum(1 for event in events if event.event_type == key) for key in sorted({event.event_type for event in events})},
    }


@router.get("/flying-objects")
def flying_objects():
    return {
        "model": video_processor.pipeline.drone_detector.status(),
        "results": [
            {
                "camera_id": camera_id,
                "label": detection.label,
                "confidence": round(detection.confidence, 4),
                "bbox": list(detection.bbox),
            }
            for camera_id, items in video_processor.pipeline.last_flying_detections.items()
            for detection in items
        ],
    }
