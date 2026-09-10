from __future__ import annotations

from typing import Any


class FaceDetectionService:
    """Local face detection adapter. Recognition is intentionally a separate, opt-in integration."""
    def __init__(self) -> None:
        self._classifier: Any | None = None

    def _load(self):
        if self._classifier is None:
            import cv2
            path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self._classifier = cv2.CascadeClassifier(path)
        return self._classifier

    def detect(self, frame: Any) -> list[tuple[int, int, int, int]]:
        import cv2
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._load().detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [tuple(int(v) for v in face) for face in faces]
