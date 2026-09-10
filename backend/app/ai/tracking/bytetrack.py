from __future__ import annotations

from typing import Any

from app.ai.interfaces import Detection, Track, Tracker


class ByteTrackTracker(Tracker):
    """Ultralytics-backed tracker adapter using ByteTrack IDs."""

    def __init__(self, model_path: str = "yolo11n.pt", confidence: float = 0.35) -> None:
        self.model_path = model_path
        self.confidence = confidence
        self._model: Any | None = None

    def update(self, detections: list[Detection], frame: Any) -> list[Track]:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.model_path)
        results = self._model.track(source=frame, conf=self.confidence, persist=True, tracker="bytetrack.yaml", verbose=False)[0]
        if results.boxes is None:
            return []
        names = results.names
        tracks: list[Track] = []
        ids = results.boxes.id
        if ids is None:
            return tracks
        for box, conf, cls, track_id in zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls, ids):
            x1, y1, x2, y2 = (float(v) for v in box.tolist())
            tracks.append(Track(
                track_id=int(track_id), label=str(names[int(cls)]), confidence=float(conf),
                bbox=(x1, y1, x2, y2), center=((x1 + x2) / 2, (y1 + y2) / 2),
            ))
        return tracks
