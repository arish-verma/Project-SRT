from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["name"] == "Project SRT"


def test_health():
    response = client.get("/api/v1/system/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_camera_lifecycle():
    camera = {
        "camera_id": "TEST-01",
        "name": "Test Camera",
        "source": "sample.mp4",
        "location": "test",
        "enabled": True,
    }
    created = client.post("/api/v1/cameras", json=camera)
    assert created.status_code == 201
    fetched = client.get("/api/v1/cameras/TEST-01")
    assert fetched.status_code == 200
    assert fetched.json()["source"] == "sample.mp4"
