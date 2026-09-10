from fastapi.testclient import TestClient

from app.main import app
from app.api.cameras import camera_manager


def setup_function() -> None:
    for camera in camera_manager.list():
        camera_manager.delete(camera.camera_id)


def test_camera_crud_and_source_type() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/cameras", json={
            "name": "North Gate",
            "source": "rtsp://example.local/live",
            "location": "North Gate",
        })
        assert response.status_code == 201
        camera = response.json()
        assert camera["source_type"] == "RTSP"
        camera_id = camera["camera_id"]

        response = client.get(f"/api/v1/cameras/{camera_id}")
        assert response.status_code == 200
        assert response.json()["name"] == "North Gate"

        response = client.patch(f"/api/v1/cameras/{camera_id}", json={"name": "North Gate A"})
        assert response.status_code == 200
        assert response.json()["name"] == "North Gate A"

        response = client.delete(f"/api/v1/cameras/{camera_id}")
        assert response.status_code == 204
        assert client.get(f"/api/v1/cameras/{camera_id}").status_code == 404


def test_camera_not_found() -> None:
    with TestClient(app) as client:
        assert client.get("/api/v1/cameras/does-not-exist").status_code == 404
        assert client.patch("/api/v1/cameras/does-not-exist", json={"name": "x"}).status_code == 404
        assert client.delete("/api/v1/cameras/does-not-exist").status_code == 404
