from fastapi.testclient import TestClient

from src.api.main import app

client = TestClient(app)


def test_health_is_always_ok_even_without_dependencies():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_ready_reports_each_dependency(live_stack):
    resp = client.get("/ready")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["checks"]["qdrant"] == "ok"
    assert body["checks"]["redis"] == "ok"
    assert body["checks"]["rerank_service"] == "ok"


def test_every_response_carries_a_request_id_header():
    resp = client.get("/health")
    assert "X-Request-Id" in resp.headers
