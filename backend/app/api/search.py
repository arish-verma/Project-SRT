import re
from fastapi import APIRouter,Query
from app.services.runtime import event_store,camera_manager
router=APIRouter(prefix='/search',tags=['search'])
def parse_query(q):
 text=q.lower().strip();f={}
 if any(w in text for w in ('person','people','human')):f['object_type']='person'
 elif 'drone' in text:f['object_type']='drone'
 elif any(w in text for w in ('vehicle','car','truck','bus','motorcycle')):f['vehicle']=True
 if any(w in text for w in ('restricted','intrusion','fence')):f['event_type']='INTRUSION'
 elif any(w in text for w in ('loiter','dwell')):f['event_type']='LOITERING'
 elif any(w in text for w in ('night','night-time','nighttime')):f['event_type']='NIGHT_MOVEMENT'
 elif any(w in text for w in ('fight','fighting','altercation','physical')):f['event_type']='FIGHT_SUSPECTED'
 elif 'drone' in text:f['event_type']='DRONE_DETECTED'
 elif any(w in text for w in ('multiple people','group','crowd')):f['event_type']='MULTI_PERSON_ACTIVITY'
 elif any(w in text for w in ('anomaly','unusual','suspicious')):f['event_type']='ANOMALY_SUSPECTED'
 m=re.search(r'camera\s*(\d+)',text)
 if m:
  n=int(m.group(1));cams=camera_manager.list()
  if 1<=n<=len(cams):f['camera_id']=cams[n-1].camera_id
 if any(w in text for w in ('high risk','critical','danger','serious')):f['min_risk']=70
 elif 'medium risk' in text:f['min_risk']=45
 elif 'low risk' in text:f['min_risk']=20
 return f
@router.get('')
def natural_language_search(q:str=Query(min_length=1,max_length=300),limit:int=100):
 f=parse_query(q);events=event_store.list(camera_id=f.get('camera_id'),event_type=f.get('event_type'),limit=min(limit,500))
 if f.get('object_type'):events=[e for e in events if e.object_type==f['object_type']]
 if f.get('vehicle'):events=[e for e in events if e.object_type in {'car','truck','bus','motorcycle'}]
 if f.get('min_risk'):events=[e for e in events if e.risk_score>=f['min_risk']]
 return {'query':q,'interpreted_filters':f,'results':[e.model_dump(mode='json')|{'evidence_url':f'/api/v1/events/evidence/{e.event_id}' if e.evidence_frame else None} for e in events]}
