"""Integration tests for FastAPI endpoints."""
from __future__ import annotations

from fastapi.testclient import TestClient

from plantguard.src.backend.main import app


client = TestClient(app)


def test_healthcheck():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
