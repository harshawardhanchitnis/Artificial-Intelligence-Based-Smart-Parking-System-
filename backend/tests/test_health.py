from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "healthy"
    assert payload["database"] == "connected"


def test_readiness_endpoint_returns_structured_checks() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health/ready")

    assert response.status_code in {200, 503}
    payload = response.json()
    assert payload["status"] in {"ready", "not_ready"}
    assert payload["summary"]["total"] == 9
    assert len(payload["checks"]) == 9


def test_system_endpoint_confirms_offline_scope() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/system")

    assert response.status_code == 200
    payload = response.json()
    assert payload["live_data_enabled"] is False
    assert payload["cloud_ai_enabled"] is False
    assert payload["local_ai_enabled"] is True
