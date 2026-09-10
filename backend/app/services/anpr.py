from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class PlateRead:
    text: str
    confidence: float
    bbox: tuple[float, float, float, float]


class ANPRService:
    """Optional ANPR boundary. OCR is deliberately injectable so deployments can use PaddleOCR/EasyOCR later."""
    PLATE_RE = re.compile(r"^[A-Z0-9 -]{4,12}$")

    def __init__(self, ocr: Any | None = None) -> None:
        self.ocr = ocr

    def read(self, crop: Any) -> PlateRead | None:
        if self.ocr is None:
            return None
        result = self.ocr(crop)
        if not result:
            return None
        text, confidence, bbox = result
        text = re.sub(r"[^A-Z0-9 -]", "", str(text).upper()).strip()
        if not self.PLATE_RE.match(text):
            return None
        return PlateRead(text, float(confidence), tuple(float(x) for x in bbox))
