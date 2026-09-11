from __future__ import annotations

import re
from typing import Any

import cv2
import numpy as np

try:
    import pytesseract
except Exception:  # optional dependency
    pytesseract = None

VEHICLES={"car","truck","bus","motorcycle"}

class ANPREngine:
    """Lightweight ANPR layer. Uses vehicle detections and optional Tesseract OCR.
    It is intentionally conservative: unreadable text is returned as a plate candidate,
    never as a fabricated registration number.
    """
    def __init__(self, enabled: bool=True): self.enabled=enabled

    def scan(self, frame: np.ndarray, detections: list[Any]) -> list[dict]:
        if not self.enabled: return []
        results=[]
        for d in detections:
            if d.label.lower() not in VEHICLES: continue
            x1,y1,x2,y2=map(int,d.bbox); h,w=frame.shape[:2]
            x1=max(0,x1); y1=max(0,y1); x2=min(w,x2); y2=min(h,y2)
            if x2-x1<30 or y2-y1<20: continue
            crop=frame[y1:y2,x1:x2]
            text=self._ocr_vehicle(crop)
            results.append({"vehicle_type":d.label,"confidence":round(float(d.confidence),3),"bbox":[x1,y1,x2,y2],"plate_text":text,"status":"READ" if text else "PLATE_CANDIDATE"})
        return results

    def _ocr_vehicle(self,crop:np.ndarray)->str|None:
        if pytesseract is None: return None
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        gray=cv2.resize(gray,None,fx=2.5,fy=2.5,interpolation=cv2.INTER_CUBIC)
        candidates=[gray,cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]]
        for img in candidates:
            try:
                raw=pytesseract.image_to_string(img,config="--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                cleaned=re.sub(r"[^A-Z0-9]","",raw.upper())
                if 4<=len(cleaned)<=12: return cleaned
            except Exception: return None
        return None
