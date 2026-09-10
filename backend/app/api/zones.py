from fastapi import APIRouter, HTTPException, status

from app.schemas.zone import Zone, ZoneCreate
from app.services.zone_manager import ZoneManager

router = APIRouter(prefix="/zones", tags=["zones"])
zone_manager = ZoneManager()


@router.get("", response_model=list[Zone])
def list_zones(camera_id: str | None = None) -> list[Zone]:
    return zone_manager.list(camera_id)


@router.post("", response_model=Zone, status_code=status.HTTP_201_CREATED)
def create_zone(data: ZoneCreate) -> Zone:
    return zone_manager.create(data)


@router.delete("/{zone_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_zone(zone_id: str) -> None:
    if not zone_manager.delete(zone_id):
        raise HTTPException(status_code=404, detail="Zone not found")
