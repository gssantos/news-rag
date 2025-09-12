import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    """Fixture providing a TestClient with automatic dependency override cleanup."""
    yield TestClient(app)
    # Cleanup after each test
    app.dependency_overrides.clear()


class TestSystemEndpoints:
    """System endpoint tests (tests 47-48)"""

    def test_get_healthz_returns_ok(self, client):
        """Test 47: GET /healthz returns ok"""
        response = client.get("/healthz")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_get_root_returns_message(self, client):
        """Test 48: GET / returns message"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "News RAG API" in data["message"]
