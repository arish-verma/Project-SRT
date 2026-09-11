from fastapi import APIRouter,HTTPException
from app.schemas.alert import AlertRecord,AlertStatus,AlertUpdate
from app.services.runtime import alert_store
router=APIRouter(prefix='/alerts',tags=['alerts'])
@router.get('',response_model=list[AlertRecord])
def list_alerts(status:AlertStatus|None=None,limit:int=100):return alert_store.list(status=status,limit=limit)
@router.patch('/{alert_id}',response_model=AlertRecord)
def update_alert(alert_id:str,payload:AlertUpdate):
 a=alert_store.update(alert_id,payload.status)
 if not a:raise HTTPException(404,'Alert not found')
 return a
@router.delete('/all',status_code=204)
def clear_alerts():alert_store.clear()
@router.delete('/{alert_id}',status_code=204)
def delete_alert(alert_id:str):
 if not alert_store.delete(alert_id):raise HTTPException(404,'Alert not found')
