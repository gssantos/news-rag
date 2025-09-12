import os
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, List
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient


class FakeAsyncSession:
    """Fake AsyncSession with overridable methods for testing"""

    def __init__(self):
        self.execute = AsyncMock()
        self.get = AsyncMock()
        self.commit = AsyncMock()
        self.rollback = AsyncMock()
        self.add = MagicMock()
        self.merge = AsyncMock()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


class FakeResult:
    """Fake SQLAlchemy result with fetchall method"""

    def __init__(self, rows: List[Any]):
        self._rows = rows

    def fetchall(self):
        return self._rows


def create_fake_article(**overrides) -> SimpleNamespace:
    """Create a minimal Article-like object for testing"""
    defaults = {
        "id": uuid4(),
        "url": "https://example.com/article",
        "title": "Test Article",
        "summary": "Test summary",
        "content": "Test content",
        "source_domain": "example.com",
        "published_at": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "created_at": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "updated_at": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "fetched_at": datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        "llm_model": "gpt-5-mini",
        "embed_model": "text-embedding-3-small",
        "embed_dim": 1536,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


@pytest.fixture
def fake_article():
    """Fixture providing a fake article object"""
    return create_fake_article()


@pytest.fixture
def fake_async_session():
    """Fixture providing a fake AsyncSession"""
    return FakeAsyncSession()


@pytest.fixture
def override_get_db_session(fake_async_session):
    """Fixture to override get_db_session dependency"""

    async def _get_db_session():
        yield fake_async_session

    return _get_db_session


@pytest.fixture
def override_get_api_key():
    """Fixture to override get_api_key dependency (returns None for no auth)"""

    def _get_api_key():
        return None

    return _get_api_key


@pytest.fixture
def fake_ingestion_service():
    """Fixture providing a mock ingestion service"""
    service = MagicMock()
    service.ingest_url = AsyncMock()
    return service


@pytest.fixture
def fake_search_service():
    """Fixture providing a mock search service"""
    service = MagicMock()
    service.search = AsyncMock()
    return service


@pytest.fixture
def fake_llm_service():
    """Fixture providing a mock LLM service"""
    service = MagicMock()
    service.generate_summary = AsyncMock()
    service.generate_embedding = AsyncMock()
    service.llm_model_name = "gpt-5-mini"
    service.emb_model_name = "text-embedding-3-small"
    service.emb_dim = 1536
    return service


@pytest.fixture
def test_client():
    """Fixture providing a FastAPI test client"""
    from app.main import app

    return TestClient(app)


@pytest.fixture
def async_client():
    """Fixture providing an async HTTP client"""
    import httpx

    return httpx.AsyncClient()


@pytest.fixture(autouse=True)
def clean_env():
    """Clean environment variables before each test"""
    original_env = dict(os.environ)
    yield
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def mock_settings():
    """Fixture providing mock settings with safe defaults"""
    from app.core.config import Settings

    return Settings(
        DATABASE_URL="postgresql+asyncpg://test:test@localhost:5432/test",
        OPENAI_API_KEY="test-key",
        API_KEY=None,
        ALLOWED_DOMAINS_STR=None,
        LOG_LEVEL="DEBUG",
    )


# Event loop fixture removed - using pytest-asyncio default
# The deprecated custom event_loop fixture has been removed
