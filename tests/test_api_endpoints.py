from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.endpoints import get_ingestion_service, get_search_service
from app.core.security import get_api_key
from app.db.session import get_db_session
from app.main import app
from tests.conftest import create_fake_article


@pytest.fixture
def client():
    """Fixture providing a TestClient with automatic dependency override cleanup."""
    yield TestClient(app)
    # Cleanup after each test
    app.dependency_overrides.clear()


@pytest.fixture
def no_api_key():
    """Fixture that overrides API key requirement (returns None)."""

    def override_get_api_key():
        return None

    app.dependency_overrides[get_api_key] = override_get_api_key
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_ingestion_service_existing():
    """Fixture for ingestion service that returns existing article."""
    fake_article = create_fake_article()
    mock_service = Mock()
    mock_service.ingest_url = AsyncMock(return_value=(fake_article, "existing"))

    def override_get_ingestion_service():
        return mock_service

    app.dependency_overrides[get_ingestion_service] = override_get_ingestion_service
    app.dependency_overrides[get_api_key] = lambda: None
    yield mock_service
    app.dependency_overrides.clear()


@pytest.fixture
def mock_ingestion_service_created():
    """Fixture for ingestion service that returns created article."""
    fake_article = create_fake_article()
    mock_service = Mock()
    mock_service.ingest_url = AsyncMock(return_value=(fake_article, "created"))

    def override_get_ingestion_service():
        return mock_service

    app.dependency_overrides[get_ingestion_service] = override_get_ingestion_service
    app.dependency_overrides[get_api_key] = lambda: None
    yield mock_service
    app.dependency_overrides.clear()


@pytest.fixture
def mock_ingestion_service_422():
    """Fixture for ingestion service that raises 422 HTTPException."""
    mock_service = Mock()
    mock_service.ingest_url = AsyncMock(
        side_effect=HTTPException(status_code=422, detail="Failed to extract content")
    )

    def override_get_ingestion_service():
        return mock_service

    app.dependency_overrides[get_ingestion_service] = override_get_ingestion_service
    app.dependency_overrides[get_api_key] = lambda: None
    yield mock_service
    app.dependency_overrides.clear()


@pytest.fixture
def mock_db_session_with_article():
    """Fixture for DB session that returns a fake article."""
    fake_article = create_fake_article()
    fake_session = Mock()
    fake_session.get = AsyncMock(return_value=fake_article)

    async def override_get_db_session():
        yield fake_session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_api_key] = lambda: None
    yield fake_session
    app.dependency_overrides.clear()


@pytest.fixture
def mock_db_session_empty():
    """Fixture for DB session that returns None (article not found)."""
    fake_session = Mock()
    fake_session.get = AsyncMock(return_value=None)

    async def override_get_db_session():
        yield fake_session

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_api_key] = lambda: None
    yield fake_session
    app.dependency_overrides.clear()


@pytest.fixture
def mock_search_service():
    """Fixture for search service with mock results."""
    fake_results = [
        SimpleNamespace(
            id=uuid4(),
            url="https://example.com/1",
            title="Article 1",
            summary="Summary 1",
            published_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
            score=0.9,
        ),
        SimpleNamespace(
            id=uuid4(),
            url="https://example.com/2",
            title="Article 2",
            summary="Summary 2",
            published_at=datetime(2023, 1, 2, tzinfo=timezone.utc),
            score=0.8,
        ),
    ]

    mock_service = Mock()
    mock_service.search = AsyncMock(return_value=fake_results)

    def override_get_search_service():
        return mock_service

    app.dependency_overrides[get_search_service] = override_get_search_service
    app.dependency_overrides[get_api_key] = lambda: None
    yield mock_service
    app.dependency_overrides.clear()


@pytest.fixture
def mock_api_key_error():
    """Fixture that makes API key dependency raise an unhandled error."""

    def override_get_api_key():
        raise ValueError("Unhandled error")

    app.dependency_overrides[get_api_key] = override_get_api_key
    yield
    app.dependency_overrides.clear()


@pytest.fixture
def mock_api_key_403():
    """Fixture that makes API key dependency raise a 403 HTTPException."""

    def override_get_api_key():
        raise HTTPException(status_code=403, detail="Could not validate credentials")

    app.dependency_overrides[get_api_key] = override_get_api_key
    yield
    app.dependency_overrides.clear()


class TestAPIEndpoints:
    """API endpoint tests (tests 36-45)"""

    def test_post_ingest_url_returns_200_for_existing(
        self, client, mock_ingestion_service_existing
    ):
        """Test 36: POST /api/v1/ingest/url returns 200 for existing"""
        response = client.post(
            "/api/v1/ingest/url",
            json={"url": "https://example.com/article", "force": False},
        )

        assert response.status_code == 200
        data = response.json()
        assert "title" in data
        assert "summary" in data
        assert "X-Request-ID" in response.headers

    def test_post_ingest_url_returns_201_for_created_updated(
        self, client, mock_ingestion_service_created
    ):
        """Test 37: POST /api/v1/ingest/url returns 201 for created/updated"""
        response = client.post(
            "/api/v1/ingest/url",
            json={"url": "https://example.com/article", "force": False},
        )

        assert response.status_code == 201
        data = response.json()
        assert "title" in data
        assert "summary" in data

    def test_post_ingest_url_propagates_422_via_middleware_shape(
        self, client, mock_ingestion_service_422
    ):
        """Test 38: POST /api/v1/ingest/url propagates 422 via middleware shape"""
        response = client.post(
            "/api/v1/ingest/url",
            json={"url": "https://example.com/article", "force": False},
        )

        assert response.status_code == 422
        data = response.json()
        # Assert standardized error format
        assert data["error"] == "http_422"
        assert data["message"] == "Failed to extract content"
        assert "details" in data

    def test_get_content_id_returns_article_detail(
        self, client, mock_db_session_with_article
    ):
        """Test 39: GET /api/v1/content/{id} returns article detail"""
        # Get the fake article from the mock session
        fake_article = create_fake_article()

        response = client.get(f"/api/v1/content/{fake_article.id}")

        assert response.status_code == 200
        data = response.json()
        assert "source_domain" in data
        assert "fetched_at" in data
        assert "content" in data
        assert "llm_model" in data
        assert "embed_model" in data
        assert "embed_dim" in data

    def test_get_content_id_returns_404_with_standardized_error_shape(
        self, client, mock_db_session_empty
    ):
        """Test 40: GET /api/v1/content/{id} returns 404 with standardized error shape"""
        article_id = uuid4()
        response = client.get(f"/api/v1/content/{article_id}")

        assert response.status_code == 404
        data = response.json()
        # Assert standardized error format
        assert data["error"] == "http_404"
        assert data["message"] == "Article not found."
        assert "details" in data

    def test_get_content_malformed_yields_standardized_validation_error(
        self, client, no_api_key
    ):
        """Test 41: GET /api/v1/content/{malformed} yields standardized validation error"""
        response = client.get("/api/v1/content/not-a-uuid")

        # FastAPI returns 422 for validation errors, now standardized by our exception handler
        assert response.status_code == 422
        data = response.json()
        # Assert standardized validation error format
        assert data["error"] == "validation_error"
        assert data["message"] == "Request validation failed"
        assert "details" in data
        assert isinstance(data["details"], list)

    def test_get_search_enforces_start_date_le_end_date(self, client, no_api_key):
        """Test 42: GET /api/v1/search enforces start_date <= end_date"""
        response = client.get(
            "/api/v1/search",
            params={
                "query": "test",
                "start_date": "2023-12-31T23:59:59Z",
                "end_date": "2023-01-01T00:00:00Z",
            },
        )

        assert response.status_code == 400
        data = response.json()
        # Assert standardized error format
        assert data["error"] == "http_400"
        assert "start_date must be before end_date" in data["message"]
        assert "details" in data

    def test_get_search_validates_k_bounds(self, client, no_api_key):
        """Test 43: GET /api/v1/search validates k bounds"""
        response = client.get("/api/v1/search", params={"query": "test", "k": 0})

        # FastAPI should validate k >= 1
        assert response.status_code == 422

    def test_get_search_returns_hits(self, client, mock_search_service):
        """Test 44: GET /api/v1/search returns hits"""
        response = client.get("/api/v1/search", params={"query": "test"})

        assert response.status_code == 200
        data = response.json()
        assert "hits" in data
        assert len(data["hits"]) == 2
        assert "X-Request-ID" in response.headers

    def test_middleware_sets_request_id_header_and_logs_unhandled_errors_as_500(
        self, client, mock_api_key_error
    ):
        """Test 45: Middleware sets X-Request-ID header and logs unhandled errors as 500 with standardized shape"""
        response = client.get("/api/v1/search", params={"query": "test"})

        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "internal_error"
        assert data["message"] == "An unexpected error occurred."
        assert "details" in data
        assert "X-Request-ID" in response.headers

    def test_api_key_required_when_configured(self, client, mock_api_key_403):
        """Test 46: API key required when configured"""
        # Test without API key
        response = client.get("/api/v1/search", params={"query": "test"})
        assert response.status_code == 403

        # Note: Testing with correct API key would require more complex mocking
        # as the API key validation is integrated with the security module
