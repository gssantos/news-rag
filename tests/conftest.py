import os
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import Any, Callable, List
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
        self.refresh = AsyncMock()
        self.delete = AsyncMock()
        self.close = AsyncMock()

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
async def async_client():
    """Fixture providing an async HTTP client"""
    import httpx

    async with httpx.AsyncClient() as client:
        yield client


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


# -----------------------
# Additional helpers/fixtures for topic API and scheduler tests
# -----------------------


@pytest.fixture
def api_client(override_get_db_session, override_get_api_key):
    """FastAPI TestClient with dependency overrides for DB and API key."""
    from app.core.security import get_api_key
    from app.db.session import get_db_session
    from app.main import app

    app.dependency_overrides[get_db_session] = override_get_db_session
    app.dependency_overrides[get_api_key] = override_get_api_key
    client = TestClient(app)
    try:
        yield client
    finally:
        app.dependency_overrides.clear()


@pytest.fixture
def mock_scheduler_endpoints(monkeypatch):
    """Monkeypatch the scheduler used by topic endpoints to a simple stub."""
    from app.api import topic_endpoints

    stub = MagicMock()
    stub.schedule_topic = MagicMock()
    stub.remove_topic = MagicMock()
    stub.update_topic_schedule = MagicMock()
    stub.pause_topic = MagicMock()
    stub.resume_topic = MagicMock()

    monkeypatch.setattr(topic_endpoints, "scheduler", stub)
    return stub


@pytest.fixture
def make_result_scalar_one() -> Callable[[Any], MagicMock]:
    """Factory to create a SQLAlchemy-like result with scalar_one_or_none."""

    def _make(obj: Any) -> MagicMock:
        result = MagicMock()
        result.scalar_one_or_none.return_value = obj
        return result

    return _make


@pytest.fixture
def make_result_scalars_all() -> Callable[[List[Any]], MagicMock]:
    """Factory to create a SQLAlchemy-like result with scalars().all()."""

    def _make(items: List[Any]) -> MagicMock:
        result = MagicMock()
        scalars = MagicMock()
        scalars.all.return_value = items
        result.scalars.return_value = scalars
        return result

    return _make


@pytest.fixture
def topic_factory():
    """Factory to create Topic objects with sensible defaults."""
    from uuid import uuid4

    from app.models.topic import Topic

    def _make(**overrides):
        now = datetime.now(timezone.utc)
        defaults = {
            "id": uuid4(),
            "name": "AI News",
            "slug": "ai-news",
            "description": "All about AI",
            "keywords": ["ai", "ml"],
            "is_active": True,
            "schedule_interval_minutes": 60,
            "created_at": now,
            "updated_at": now,
            "last_ingestion_at": None,
            "sources": [],
            "ingestion_runs": [],
        }
        defaults.update(overrides)
        topic = Topic(
            name=defaults["name"],
            slug=defaults["slug"],
            description=defaults["description"],
            keywords=defaults["keywords"],
            is_active=defaults["is_active"],
            schedule_interval_minutes=defaults["schedule_interval_minutes"],
        )
        # Inject fields typically set by DB/ORM
        topic.id = defaults["id"]
        topic.created_at = defaults["created_at"]
        topic.updated_at = defaults["updated_at"]
        topic.last_ingestion_at = defaults["last_ingestion_at"]
        topic.sources = defaults["sources"]
        topic.ingestion_runs = defaults["ingestion_runs"]
        return topic

    return _make


@pytest.fixture
def source_factory():
    """Factory to create NewsSource objects with sensible defaults."""
    from uuid import uuid4

    from app.models.topic import NewsSource, SourceType

    def _make(**overrides):
        now = datetime.now(timezone.utc)
        defaults = {
            "id": uuid4(),
            "topic_id": uuid4(),
            "name": "Example RSS",
            "url": "https://example.com/rss.xml",
            "source_type": SourceType.RSS,
            "config": {},
            "is_active": True,
            "last_successful_fetch": None,
            "consecutive_failures": 0,
            "created_at": now,
            "updated_at": now,
        }
        defaults.update(overrides)
        src = NewsSource(
            topic_id=defaults["topic_id"],
            name=defaults["name"],
            url=defaults["url"],
            source_type=defaults["source_type"],
            config=defaults["config"],
            is_active=defaults["is_active"],
        )
        src.id = defaults["id"]
        src.last_successful_fetch = defaults["last_successful_fetch"]
        src.consecutive_failures = defaults["consecutive_failures"]
        src.created_at = defaults["created_at"]
        src.updated_at = defaults["updated_at"]
        return src

    return _make


@pytest.fixture
def ingestion_run_factory():
    """Factory to create IngestionRun-like objects with sensible defaults."""
    from uuid import uuid4

    from app.models.topic import IngestionRun, IngestionStatus

    def _make(**overrides):
        now = datetime.now(timezone.utc)
        defaults = {
            "id": uuid4(),
            "topic_id": uuid4(),
            "status": IngestionStatus.PENDING,
            "articles_discovered": 0,
            "articles_ingested": 0,
            "articles_failed": 0,
            "articles_duplicates": 0,
            "started_at": now,
            "completed_at": None,
            "error_messages": [],
        }
        defaults.update(overrides)
        run = IngestionRun(
            topic_id=defaults["topic_id"],
            status=defaults["status"],
            articles_discovered=defaults["articles_discovered"],
            articles_ingested=defaults["articles_ingested"],
            articles_failed=defaults["articles_failed"],
            articles_duplicates=defaults["articles_duplicates"],
            started_at=defaults["started_at"],
            completed_at=defaults["completed_at"],
            error_messages=defaults["error_messages"],
        )
        run.id = defaults["id"]
        return run

    return _make
