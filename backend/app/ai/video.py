from __future__ import annotations

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

    def _open(self) -> cv2.VideoCapture:
        # DirectShow is generally the most reliable Windows backend for a local
        # webcam. Do not force it for files/RTSP because those are FFmpeg-backed.
        if isinstance(self.source, int):
            capture = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
            if not capture.isOpened():
                capture.release()
                capture = cv2.VideoCapture(self.source)
        else:
            capture = cv2.VideoCapture(self.source)

        if isinstance(self.source, int):
            # Not every backend honors this property; setting it is harmless and
            # helps prevent stale webcam frames where supported.
            capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def frames(self) -> Iterator[FramePacket]:
        self.capture = self._open()
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source_id} ({self.source})")

        index = 0
        try:
            while True:
                ok, frame = self.capture.read()
                if not ok or frame is None or frame.size == 0:
                    break
                yield FramePacket(
                    frame=frame,
                    timestamp=time.time(),
                    source_id=self.source_id,
                    frame_index=index,
                )
                index += 1
        finally:
            self.close()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
