from fastapi import APIRouter, HTTPException, status

from app.schemas.zone import Zone, ZoneCreate
from app.services.runtime import zone_manager

router = APIRouter(prefix="/zones", tags=["zones"])


@router.get("", response_model=list[Zone])
def list_zones(camera_id: str | None = None) -> list[Zone]:
    return zone_manager.list(camera_id)


@router.post("", response_model=Zone, status_code=status.HTTP_201_CREATED)
def create_zone(data: ZoneCreate) -> Zone:
    if any(not (0 <= x <= 1 and 0 <= y <= 1) for x, y in data.polygon):
        raise HTTPException(status_code=400, detail="Polygon coordinates must be normalized between 0 and 1")
    return zone_manager.create(data)


@router.delete("/{zone_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_zone(zone_id: str) -> None:
    if not zone_manager.delete(zone_id):
        raise HTTPException(status_code=404, detail="Zone not found")
