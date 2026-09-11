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
 def __init__(self,enabled:bool=True):self.enabled=enabled
 def scan(self,frame:np.ndarray,detections:list[Any],camera_id:str|None=None,timestamp:float|None=None)->list[dict]:
  if not self.enabled:return []
  results=[];ts=datetime.fromtimestamp(timestamp or datetime.now(tz=timezone.utc).timestamp(),tz=timezone.utc).isoformat()
  for d in detections:
   if d.label.lower() not in VEHICLES:continue
   x1,y1,x2,y2=map(int,d.bbox);h,w=frame.shape[:2];x1,y1,x2,y2=max(0,x1),max(0,y1),min(w,x2),min(h,y2)
   if x2-x1<30 or y2-y1<20:continue
   crop=frame[y1:y2,x1:x2];text,plate_box=self._read_plate(crop)
   results.append({'vehicle_type':d.label,'confidence':round(float(d.confidence),3),'bbox':[x1,y1,x2,y2],'plate_bbox':([x1+plate_box[0],y1+plate_box[1],x1+plate_box[2],y1+plate_box[3]] if plate_box else None),'plate_text':text,'status':'READ' if text else 'PLATE_CANDIDATE','camera_id':camera_id,'timestamp':ts})
  return results
 def _read_plate(self,crop:np.ndarray):
  if pytesseract is None:return None,None
  h,w=crop.shape[:2];regions=[]
  for y0,y1 in ((.35,1.),(.50,1.),(.25,.85)):
   yy0,yy1=int(h*y0),int(h*y1);roi=crop[yy0:yy1,:]
   if roi.size:regions.append((roi,(0,yy0,w,yy1)))
  gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY);kernel=cv2.getStructuringElement(cv2.MORPH_RECT,(max(9,w//8),max(3,h//12)));blackhat=cv2.morphologyEx(gray,cv2.MORPH_BLACKHAT,kernel);_,th=cv2.threshold(blackhat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU);contours,_=cv2.findContours(th,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  for c in contours:
   rx,ry,rw,rh=cv2.boundingRect(c);ratio=rw/max(rh,1);area=rw*rh
   if 2<=ratio<=7.5 and area>=max(80,w*h*.008) and rw>=.15*w:
    px,py=max(2,int(rw*.08)),max(2,int(rh*.25));ax1,ay1=max(0,rx-px),max(0,ry-py);ax2,ay2=min(w,rx+rw+px),min(h,ry+rh+py);regions.append((crop[ay1:ay2,ax1:ax2],(ax1,ay1,ax2,ay2)))
  for roi,box in regions:
   text=self._ocr_roi(roi)
   if text:return text,box
  return None,None
 def _ocr_roi(self,roi):
  if roi is None or roi.size==0 or min(roi.shape[:2])<8:return None
  gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY);gray=cv2.resize(gray,None,fx=4.,fy=4.,interpolation=cv2.INTER_CUBIC);variants=[gray,cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1],cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,9)]
  for img in variants:
   for psm in (7,8,6):
    try:raw=pytesseract.image_to_string(img,config=f'--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    except Exception:continue
    cleaned=re.sub(r'[^A-Z0-9]','',raw.upper())
    if 4<=len(cleaned)<=12 and any(c.isdigit() for c in cleaned) and any(c.isalpha() for c in cleaned) and (len(cleaned)<=10 or PLATE_RE.match(cleaned)):return cleaned
  return None
