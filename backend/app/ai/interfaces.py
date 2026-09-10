from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Iterator

@dataclass
class FramePacket:
    frame: Any
    timestamp: float
    source_id: str
    frame_index: int

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]

@dataclass
class Track:
    track_id: int
    label: str
    confidence: float
    bbox: tuple[float, float, float, float]
    center: tuple[float, float]

class VideoSource(ABC):
    @abstractmethod
    def frames(self) -> Iterator[FramePacket]: ...

    @abstractmethod
    def close(self) -> None: ...

class Detector(ABC):
    @abstractmethod
    def detect(self, frame: Any) -> list[Detection]: ...

class Tracker(ABC):
    @abstractmethod
    def update(self, detections: list[Detection], frame: Any) -> list[Track]: ...

class EventEngine(ABC):
    @abstractmethod
    def evaluate(self, tracks: list[Track], context: dict[str, Any]) -> list[Any]: ...
