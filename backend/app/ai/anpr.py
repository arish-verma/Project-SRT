from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any

import cv2
import numpy as np

try:
    import pytesseract
except Exception:
    pytesseract = None

VEHICLES = {"car", "truck", "bus", "motorcycle"}
PLATE_RE = re.compile(r"^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}[0-9]{1,4}$")


class ANPREngine:
    """Practical ANPR for ordinary CCTV frames.

    It first searches likely plate regions inside each vehicle crop, then runs
    several OCR preprocessing variants. OCR is conservative: text is emitted
    only when it looks like a plausible registration string.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def scan(self, frame: np.ndarray, detections: list[Any], camera_id: str | None = None, timestamp: float | None = None) -> list[dict]:
        if not self.enabled:
            return []
        results: list[dict] = []
        ts = datetime.fromtimestamp(timestamp or datetime.now(tz=timezone.utc).timestamp(), tz=timezone.utc).isoformat()
        for d in detections:
            if d.label.lower() not in VEHICLES:
                continue
            x1, y1, x2, y2 = map(int, d.bbox)
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2 - x1 < 30 or y2 - y1 < 20:
                continue
            crop = frame[y1:y2, x1:x2]
            text, plate_box = self._read_plate(crop)
            results.append({
                "vehicle_type": d.label,
                "confidence": round(float(d.confidence), 3),
                "bbox": [x1, y1, x2, y2],
                "plate_bbox": ([x1 + plate_box[0], y1 + plate_box[1], x1 + plate_box[2], y1 + plate_box[3]] if plate_box else None),
                "plate_text": text,
                "status": "READ" if text else "PLATE_CANDIDATE",
                "camera_id": camera_id,
                "timestamp": ts,
            })
        return results

    def _read_plate(self, crop: np.ndarray) -> tuple[str | None, tuple[int, int, int, int] | None]:
        if pytesseract is None:
            return None, None
        h, w = crop.shape[:2]
        regions: list[tuple[np.ndarray, tuple[int, int, int, int]]] = []

        # Most plates occupy the lower/middle portion of a vehicle box.
        for y0, y1 in ((0.35, 1.0), (0.50, 1.0), (0.25, 0.85)):
            yy0, yy1 = int(h * y0), int(h * y1)
            roi = crop[yy0:yy1, :]
            if roi.size:
                regions.append((roi, (0, yy0, w, yy1)))

        # Add bright/dark rectangular candidates found by morphology.
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 8), max(3, h // 12))))
        _, th = cv2.threshold(blackhat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            rx, ry, rw, rh = cv2.boundingRect(c)
            ratio = rw / max(rh, 1)
            area = rw * rh
            if 2.0 <= ratio <= 7.5 and area >= max(80, w * h * 0.008) and rw >= 0.15 * w:
                pad_x, pad_y = max(2, int(rw * .08)), max(2, int(rh * .25))
                ax1, ay1 = max(0, rx - pad_x), max(0, ry - pad_y)
                ax2, ay2 = min(w, rx + rw + pad_x), min(h, ry + rh + pad_y)
                regions.append((crop[ay1:ay2, ax1:ax2], (ax1, ay1, ax2, ay2)))

        for roi, box in regions:
            text = self._ocr_roi(roi)
            if text:
                return text, box
        return None, None

    def _ocr_roi(self, roi: np.ndarray) -> str | None:
        if roi is None or roi.size == 0 or min(roi.shape[:2]) < 8:
            return None
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=4.0, fy=4.0, interpolation=cv2.INTER_CUBIC)
        variants = [
            gray,
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9),
        ]
        for img in variants:
            for psm in (7, 8, 6):
                try:
                    raw = pytesseract.image_to_string(img, config=f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                except Exception:
                    continue
                cleaned = re.sub(r"[^A-Z0-9]", "", raw.upper())
                if 4 <= len(cleaned) <= 12 and any(ch.isdigit() for ch in cleaned) and any(ch.isalpha() for ch in cleaned):
                    # Indian plates normally contain both letters and digits. Avoid returning long words.
                    if len(cleaned) <= 10 or PLATE_RE.match(cleaned):
                        return cleaned
        return None
