# Adjust the import below to match your app entrypoint
# e.g., from app.main import app  OR  from src.app import app
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_root_ok():
    r = client.get("/")
    assert r.status_code == 200
    # If your root returns JSON, optionally assert shape:
    # data = r.json()
    # assert "message" in data

def test_health_ok():
    r = client.get("/health")
    assert r.status_code == 200
    # Optionally assert response body:
    # assert r.json() == {"status": "ok"}
