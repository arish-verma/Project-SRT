from __future__ import annotations
import os,re,shutil
from collections import Counter
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

class ANPREngine:
    def __init__(self,enabled:bool=True): self.enabled=enabled

    def scan(self,frame:np.ndarray,detections:list[Any],camera_id:str|None=None,timestamp:float|None=None)->list[dict]:
        if not self.enabled:return []
        ts=datetime.fromtimestamp(timestamp or datetime.now(tz=timezone.utc).timestamp(),tz=timezone.utc).isoformat()
        results=[]
        for d in detections:
            if d.label.lower() not in VEHICLES:continue
            x1,y1,x2,y2=map(int,d.bbox);h,w=frame.shape[:2]
            x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
            result={'vehicle_type':d.label,'confidence':round(float(d.confidence),3),'bbox':[x1,y1,x2,y2],'plate_bbox':None,'plate_text':None,'status':'PLATE_CANDIDATE','camera_id':camera_id,'timestamp':ts}
            # Classification is always returned. OCR is attempted only when the vehicle
            # is large enough to contain useful plate pixels.
            if x2-x1>=35 and y2-y1>=24:
                text,plate_box=self._read_plate(frame[y1:y2,x1:x2])
                if plate_box:result['plate_bbox']=[x1+plate_box[0],y1+plate_box[1],x1+plate_box[2],y1+plate_box[3]]
                if text:result['plate_text']=text;result['status']='READ'
            results.append(result)
        return results

    def _read_plate(self,crop:np.ndarray):
        if pytesseract is None:return None,None
        h,w=crop.shape[:2]
        regions=[]
        # Search likely plate bands first. Keep the candidate count bounded for speed.
        for y0,y1 in ((.30,1.),(.45,1.),(.55,1.),(.15,.85)):
            yy0,yy1=int(h*y0),int(h*y1);roi=crop[yy0:yy1]
            if roi.size:regions.append((roi,(0,yy0,w,yy1)))
        gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
        kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(max(7,w//10),max(3,h//14)))
        blackhat=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel)
        _,th=cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        contours,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            rx,ry,rw,rh=cv2.boundingRect(c);ratio=rw/max(rh,1)
            if 2.0<=ratio<=9.0 and rw>=max(18,.12*w) and rh>=max(5,.02*h):
                pad_x=max(2,int(rw*.10));pad_y=max(2,int(rh*.45));ax1=max(0,rx-pad_x);ay1=max(0,ry-pad_y);ax2=min(w,rx+rw+pad_x);ay2=min(h,ry+rh+pad_y);regions.append((crop[ay1:ay2,ax1:ax2],(ax1,ay1,ax2,ay2)))
        # Bright-plate candidate rectangles.
        for threshold in (140,180,210):
            mask=cv2.threshold(gray,threshold,255,cv2.THRESH_BINARY)[1]
            mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(max(5,w//18),max(2,h//28))))
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for c in contours:
                rx,ry,rw,rh=cv2.boundingRect(c);ratio=rw/max(rh,1)
                if 2.0<=ratio<=8.5 and rw>=max(20,.15*w) and rh>=max(5,.018*h):
                    regions.append((crop[max(0,ry-int(.25*rh)):min(h,ry+rh+int(.25*rh)),max(0,rx-int(.05*rw)):min(w,rx+rw+int(.05*rw))],(max(0,rx-int(.05*rw)),max(0,ry-int(.25*rh)),min(w,rx+rw+int(.05*rw)),min(h,ry+rh+int(.25*rh)))))
        # Prefer smaller candidate regions (more plate-like), but cap work.
        regions=sorted(regions,key=lambda r:(r[1][2]-r[1][0])*(r[1][3]-r[1][1]))[:12]
        votes=[]
        best_box=None
        for roi,box in regions:
            for text in self._ocr_roi(roi):votes.append((text,box))
        if not votes:return None,None
        counts=Counter(t for t,_ in votes)
        text=counts.most_common(1)[0][0]
        best_box=next(b for t,b in votes if t==text)
        return text,best_box

    @staticmethod
    def _normalise_ocr(raw:str)->str:
        return re.sub(r'[^A-Z0-9]','',raw.upper())

    def _ocr_roi(self,roi):
        if roi is None or roi.size==0 or min(roi.shape[:2])<6:return []
        gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
        out=[]
        for scale in (4.,6.):
            up=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
            variants=(up,cv2.threshold(up,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1])
            for img in variants:
                for psm in (7,8):
                    try:raw=pytesseract.image_to_string(img,config=f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',timeout=1.2)
                    except Exception:continue
                    text=self._normalise_ocr(raw)
                    if self._valid_plate(text):out.append(text)
        return out

    @staticmethod
    def _valid_plate(text:str)->bool:
        if not 5<=len(text)<=12:return False
        # Accept legitimate plates with letters-only or digits-only layouts too; the
        # important requirement is enough characters to be a registration, not a word.
        if len(set(text))==1:return False
        return sum(c.isdigit() for c in text)>=2 or len(text)>=7
