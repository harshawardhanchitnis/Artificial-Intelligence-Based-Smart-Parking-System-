from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.http import RequestContextMiddleware
from app.main import app


def test_liveness_echoes_request_id_and_security_headers() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health/live",
            headers={"X-Request-ID": "reliability-test-1"},
        )

    assert response.status_code == 200
    assert response.json()["status"] == "alive"
    assert response.headers["X-Request-ID"] == "reliability-test-1"
    assert float(response.headers["X-Process-Time-Ms"]) >= 0
    assert response.headers["X-Content-Type-Options"] == "nosniff"
    assert response.headers["X-Frame-Options"] == "DENY"
    assert response.headers["Referrer-Policy"] == "no-referrer"


def test_invalid_request_id_is_replaced() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/health/live",
            headers={"X-Request-ID": "not valid because it has spaces"},
        )

    generated = response.headers["X-Request-ID"]
    assert generated != "not valid because it has spaces"
    assert len(generated) == 32


def test_validation_error_has_stable_contract_and_request_id() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/analysis/history?limit=0",
            headers={"X-Request-ID": "validation-test"},
        )

    assert response.status_code == 422
    payload = response.json()
    assert payload["detail"] == "Request validation failed"
    assert payload["error"]["code"] == "VALIDATION_ERROR"
    assert payload["error"]["request_id"] == "validation-test"
    assert payload["error"]["details"]


def test_not_found_error_has_stable_contract() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/v1/definitely-not-a-route",
            headers={"X-Request-ID": "not-found-test"},
        )

    assert response.status_code == 404
    payload = response.json()
    assert payload["detail"] == "Not Found"
    assert payload["error"] == {
        "code": "HTTP_404",
        "message": "Not Found",
        "request_id": "not-found-test",
    }


def test_unexpected_error_is_safe_and_traceable() -> None:
    isolated_app = FastAPI()
    isolated_app.add_middleware(RequestContextMiddleware)

    @isolated_app.get("/boom")
    def boom() -> None:
        raise RuntimeError("private diagnostic detail")

    with TestClient(isolated_app) as client:
        response = client.get("/boom", headers={"X-Request-ID": "unexpected-test"})

    assert response.status_code == 500
    payload = response.json()
    assert payload["detail"] == "The local service could not complete the request"
    assert payload["error"]["code"] == "INTERNAL_SERVER_ERROR"
    assert payload["error"]["request_id"] == "unexpected-test"
    assert "private diagnostic detail" not in response.text
