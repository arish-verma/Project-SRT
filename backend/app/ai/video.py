from pathlib import Path
from typing import Iterator
import time
import cv2
from .interfaces import FramePacket, VideoSource


class OpenCVVideoSource(VideoSource):
    """Local file, webcam index, or RTSP/HTTP URL source.

    Local files are paced to their native FPS so the browser sees normal-speed
    playback instead of the processor reading the entire file as fast as CPU
    and disk allow. Live sources are never artificially delayed.
    """

    def __init__(self, source: str, source_id: str = "camera-01"):
        self.source = int(source) if source.isdigit() else source
        self.source_id = source_id
        self.capture: cv2.VideoCapture | None = None

    @property
    def is_local_file(self) -> bool:
        return isinstance(self.source, str) and Path(self.source).exists()

    def frames(self) -> Iterator[FramePacket]:
        self.capture = cv2.VideoCapture(self.source)
        if not self.capture.isOpened():
            raise RuntimeError(f"Unable to open video source: {self.source_id}")

        index = 0
        fps = float(self.capture.get(cv2.CAP_PROP_FPS) or 0.0)
        fps = fps if 1.0 <= fps <= 120.0 else 0.0
        playback_started = time.monotonic()

        try:
            while True:
                # Pace local recordings to their source frame rate. This is
                # deliberately done before the next read so processing can
                # remain asynchronous without making a file finish instantly.
                if self.is_local_file and fps:
                    deadline = playback_started + (index / fps)
                    delay = deadline - time.monotonic()
                    if delay > 0:
                        time.sleep(delay)

                ok, frame = self.capture.read()
                if not ok:
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
