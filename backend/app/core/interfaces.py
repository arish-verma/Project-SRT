from abc import ABC, abstractmethod
from typing import Any, Iterable

from app.schemas.domain import Detection, Track


class VideoSource(ABC):
    @abstractmethod
    def read(self) -> tuple[bool, Any]: ...

    @abstractmethod
    def close(self) -> None: ...


class Detector(ABC):
    @abstractmethod
    def detect(self, frame: Any) -> Iterable[Detection]: ...


class Tracker(ABC):
    @abstractmethod
    def update(self, detections: Iterable[Detection]) -> Iterable[Track]: ...


class FaceDetector(ABC):
    @abstractmethod
    def detect_faces(self, frame: Any) -> Iterable[dict]: ...


class FaceRecognizer(ABC):
    @abstractmethod
    def recognize(self, face: Any) -> dict: ...


class PlateDetector(ABC):
    @abstractmethod
    def detect_plates(self, frame: Any) -> Iterable[dict]: ...


class OCR(ABC):
    @abstractmethod
    def read(self, crop: Any) -> dict: ...


class ActivityAnalyzer(ABC):
    @abstractmethod
    def analyze(self, tracks: Iterable[Track], context: dict) -> dict: ...
