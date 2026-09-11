from __future__ import annotations
import os,re,shutil
from datetime import datetime,timezone
from typing import Any
import cv2
import numpy as np
try:
    import pytesseract
    _candidates=[shutil.which('tesseract'),r'C:\Program Files\Tesseract-OCR\tesseract.exe',r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe']
    _cmd=next((p for p in _candidates if p and os.path.exists(p)),None)
    if _cmd:pytesseract.pytesseract.tesseract_cmd=_cmd
except Exception:
    pytesseract=None

VEHICLES={'car','truck','bus','motorcycle'}
PLATE_RE=re.compile(r'^[A-Z]{1,3}[0-9]{1,4}[A-Z]{0,3}[0-9]{1,4}$')

class ANPREngine:
    def __init__(self,enabled:bool=True): self.enabled=enabled

    def scan(self,frame:np.ndarray,detections:list[Any],camera_id:str|None=None,timestamp:float|None=None)->list[dict]:
        if not self.enabled:return []
        results=[]
        ts=datetime.fromtimestamp(timestamp or datetime.now(tz=timezone.utc).timestamp(),tz=timezone.utc).isoformat()
        for d in detections:
            if d.label.lower() not in VEHICLES:continue
            x1,y1,x2,y2=map(int,d.bbox);h,w=frame.shape[:2]
            x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
            if x2-x1<25 or y2-y1<18:continue
            crop=frame[y1:y2,x1:x2]
            text,plate_box=self._read_plate(crop)
            results.append({
                'vehicle_type':d.label,'confidence':round(float(d.confidence),3),
                'bbox':[x1,y1,x2,y2],
                'plate_bbox':([x1+plate_box[0],y1+plate_box[1],x1+plate_box[2],y1+plate_box[3]] if plate_box else None),
                'plate_text':text,'status':'READ' if text else 'PLATE_CANDIDATE',
                'camera_id':camera_id,'timestamp':ts
            })
        return results

    def _read_plate(self,crop:np.ndarray):
        if pytesseract is None:return None,None
        h,w=crop.shape[:2]
        regions=[]
        for y0,y1,x0,x1 in ((.25,1.,0.,1.),(.40,1.,0.,1.),(.52,1.,0.,1.),(.18,.82,.05,.95)):
            yy0,yy1=int(h*y0),int(h*y1);xx0,xx1=int(w*x0),int(w*x1)
            roi=crop[yy0:yy1,xx0:xx1]
            if roi.size:regions.append((roi,(xx0,yy0,xx1,yy1)))

        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        for source in (gray,cv2.GaussianBlur(gray,(3,3),0)):
            for threshold in (145,175,200):
                mask=cv2.threshold(source,threshold,255,cv2.THRESH_BINARY)[1]
                mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(max(3,w//20),max(2,h//35))))
                contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                for c in contours:
                    rx,ry,rw,rh=cv2.boundingRect(c);ratio=rw/max(rh,1);area=rw*rh
                    if 2.0<=ratio<=8.5 and rw>=max(30,.20*w) and rh>=max(7,.025*h) and area>=max(120,.008*w*h):
                        pad_x=max(2,int(rw*.08));pad_y=max(2,int(rh*.35))
                        ax1,ay1=max(0,rx-pad_x),max(0,ry-pad_y);ax2,ay2=min(w,rx+rw+pad_x),min(h,ry+rh+pad_y)
                        regions.append((crop[ay1:ay2,ax1:ax2],(ax1,ay1,ax2,ay2)))

        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(max(9,w//8),max(3,h//12)))
        blackhat=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
        _,th=cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            rx,ry,rw,rh=cv2.boundingRect(c);ratio=rw/max(rh,1);area=rw*rh
            if 2<=ratio<=7.5 and area>=max(80,w*h*.006) and rw>=.15*w:
                px,py=max(2,int(rw*.08)),max(2,int(rh*.30))
                ax1,ay1=max(0,rx-px),max(0,ry-py);ax2,ay2=min(w,rx+rw+px),min(h,ry+rh+py)
                regions.append((crop[ay1:ay2,ax1:ax2],(ax1,ay1,ax2,ay2)))

        seen=set()
        for roi,box in sorted(regions,key=lambda item:(item[1][2]-item[1][0])*(item[1][3]-item[1][1]),reverse=True):
            key=tuple(box)
            if key in seen:continue
            seen.add(key)
            text=self._ocr_roi(roi)
            if text:return text,box
        return None,None

    @staticmethod
    def _normalise_ocr(raw:str)->str:
        text=re.sub(r'[^A-Z0-9]','',raw.upper())
        text=text.replace(' ','')
        return text

    def _ocr_roi(self,roi):
        if roi is None or roi.size==0 or min(roi.shape[:2])<6:return None
        gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        for scale in (4.,6.,8.):
            up=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            variants=[up,cv2.threshold(up,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],cv2.adaptiveThreshold(up,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)]
            for img in variants:
                for psm in (7,8,6,13):
                    try:raw=pytesseract.image_to_string(img,config=f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
                    except Exception:continue
                    cleaned=self._normalise_ocr(raw)
                    if self._valid_plate(cleaned):return cleaned
        return None

    @staticmethod
    def _valid_plate(text:str)->bool:
        if not 6<=len(text)<=12:return False
        if not any(c.isdigit() for c in text) or not any(c.isalpha() for c in text):return False
        return bool(PLATE_RE.match(text)) or (6<=len(text)<=10 and sum(c.isdigit() for c in text)>=2)
