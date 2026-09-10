from pathlib import Path
from typing import Iterator
import time
import cv2
from .interfaces import FramePacket, VideoSource

class OpenCVVideoSource(VideoSource):
    """Local file, webcam index, or RTSP/HTTP URL source."""
    def __init__(self, source: str, source_id: str = "camera-01"):
        self.source = int(source) if source.isdigit() else source
        self.source_id = source_id
        self.capture: cv2.VideoCapture | None = None

    def frames(self) -> Iterator[FramePacket]:
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source_id}")
        index = 0
        try:
            while True:
                ok, frame = self.capture.read()
                if not ok:
                    break
                yield FramePacket(frame=frame, timestamp=time.time(), source_id=self.source_id, frame_index=index)
                index += 1
        finally:
            self.close()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
