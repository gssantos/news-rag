from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from app.models.topic import (
    IngestionRun,
    IngestionStatus,
    NewsSource,
    SourceType,
    Topic,
)
from app.services.scheduled_ingestion_service import ScheduledIngestionService


@pytest.mark.asyncio
async def test_fetch_rss_articles_success(monkeypatch):
    # Arrange: fake service with mocked HTTP client and feedparser
    service = ScheduledIngestionService()

    class FakeResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    async def fake_get(url):
        return FakeResponse(
            "<rss><channel><item><link>https://ex/1</link></item></channel></rss>"
        )

    service.http_client.get = AsyncMock(side_effect=fake_get)

    # Patch feedparser.parse to return entries with links
    fake_entry = SimpleNamespace(link="https://ex/1")

    class FakeFeed:
        entries = [fake_entry]

    import app.services.scheduled_ingestion_service as mod

    monkeypatch.setattr(
        mod, "feedparser", SimpleNamespace(parse=lambda text: FakeFeed())
    )

    src = NewsSource(
        topic_id=uuid4(),
        name="RSS",
        url="https://site/rss.xml",
        source_type=SourceType.RSS,
        config={},
        is_active=True,
    )

    # Act
    urls = await service._fetch_rss_articles(src)

    # Assert
    assert urls == ["https://ex/1"]


@pytest.mark.asyncio
async def test_ingest_topic_success_with_new_articles(fake_async_session, monkeypatch):
    # Topic with one active RSS source
    topic_id = uuid4()
    source = NewsSource(
        topic_id=topic_id,
        name="RSS",
        url="https://example.com/rss.xml",
        source_type=SourceType.RSS,
        config={},
        is_active=True,
    )
    # Initialize attributes that will be updated
    source.consecutive_failures = 0
    source.last_successful_fetch = None

    topic = Topic(
        name="AI", slug="ai", description=None, keywords=["ai"], is_active=True
    )
    topic.id = topic_id
    topic.sources = [source]

    # First execute: select topic with sources
    fake_async_session.execute.return_value = _result_scalar(topic)

    # Patch RSS fetch to return two URLs
    service = ScheduledIngestionService()
    monkeypatch.setattr(
        service,
        "_fetch_rss_articles",
        AsyncMock(return_value=["https://a/1", "https://a/2"]),
    )

    # Patch existing article check: always None (new)
    monkeypatch.setattr(
        service, "_check_existing_article", AsyncMock(return_value=None)
    )

    # Patch link to topic to avoid DB complexity
    link_mock = AsyncMock()
    monkeypatch.setattr(service, "_link_article_to_topic", link_mock)

    # Patch IngestionService used internally
    class FakeIngestionService:
        def __init__(self, db, llm_service):
            self.db = db
            self.llm_service = llm_service

        async def ingest_url(self, request):
            # Return a fake article-like object with id
            return SimpleNamespace(id=uuid4()), "ingested"

    import app.services.scheduled_ingestion_service as mod

    monkeypatch.setattr(mod, "IngestionService", FakeIngestionService)
    # LLM service
    monkeypatch.setattr(mod, "get_llm_service", lambda: MagicMock())

    run = await service.ingest_topic(str(topic_id), fake_async_session)

    assert isinstance(run, IngestionRun)
    assert run.articles_discovered == 2
    assert run.articles_ingested == 2
    assert run.articles_failed == 0
    assert run.articles_duplicates == 0
    assert run.status in (IngestionStatus.SUCCESS, str(IngestionStatus.SUCCESS))
    assert topic.last_ingestion_at is not None
    assert link_mock.await_count == 2


@pytest.mark.asyncio
async def test_ingest_topic_duplicates_only_success_status(
    fake_async_session, monkeypatch
):
    topic_id = uuid4()
    source = NewsSource(
        topic_id=topic_id,
        name="RSS",
        url="https://example.com/rss.xml",
        source_type=SourceType.RSS,
        config={},
        is_active=True,
    )
    # Initialize attributes that will be updated
    source.consecutive_failures = 0
    source.last_successful_fetch = None

    topic = Topic(
        name="AI", slug="ai", description=None, keywords=["ai"], is_active=True
    )
    topic.id = topic_id
    topic.sources = [source]

    fake_async_session.execute.return_value = _result_scalar(topic)

    service = ScheduledIngestionService()
    monkeypatch.setattr(
        service,
        "_fetch_rss_articles",
        AsyncMock(return_value=["https://a/1", "https://a/2"]),
    )
    # Check existing article returns "existing article" for both URLs
    existing_article = SimpleNamespace(id=uuid4())
    monkeypatch.setattr(
        service, "_check_existing_article", AsyncMock(return_value=existing_article)
    )
    link_mock = AsyncMock()
    monkeypatch.setattr(service, "_link_article_to_topic", link_mock)
    import app.services.scheduled_ingestion_service as mod

    monkeypatch.setattr(
        mod,
        "IngestionService",
        lambda db, llm_service: MagicMock(ingest_url=AsyncMock()),
    )  # should not be called
    monkeypatch.setattr(mod, "get_llm_service", lambda: MagicMock())

    run = await service.ingest_topic(str(topic_id), fake_async_session)

    assert run.articles_discovered == 2
    assert run.articles_ingested == 0
    assert run.articles_duplicates == 2
    assert run.articles_failed == 0
    # With no failures and no errors, code marks SUCCESS even if all duplicates
    assert run.status in (IngestionStatus.SUCCESS, str(IngestionStatus.SUCCESS))
    assert link_mock.await_count == 2


@pytest.mark.asyncio
async def test_ingest_topic_rss_fetch_error_marks_failed(
    fake_async_session, monkeypatch
):
    topic_id = uuid4()
    source = NewsSource(
        topic_id=topic_id,
        name="RSS",
        url="https://example.com/rss.xml",
        source_type=SourceType.RSS,
        config={},
        is_active=True,
    )
    # Initialize attributes that will be updated
    source.consecutive_failures = 0
    source.last_successful_fetch = None

    topic = Topic(
        name="AI", slug="ai", description=None, keywords=["ai"], is_active=True
    )
    topic.id = topic_id
    topic.sources = [source]

    fake_async_session.execute.return_value = _result_scalar(topic)

    service = ScheduledIngestionService()
    # Fetch raises an error
    monkeypatch.setattr(
        service,
        "_fetch_rss_articles",
        AsyncMock(side_effect=RuntimeError("fetch error")),
    )
    import app.services.scheduled_ingestion_service as mod

    monkeypatch.setattr(
        mod,
        "IngestionService",
        lambda db, llm_service: MagicMock(ingest_url=AsyncMock()),
    )
    monkeypatch.setattr(mod, "get_llm_service", lambda: MagicMock())

    run = await service.ingest_topic(str(topic_id), fake_async_session)

    assert run.articles_discovered == 0
    assert run.articles_ingested == 0
    assert run.articles_failed == 0  # No ingestion attempts
    assert run.status in (IngestionStatus.FAILED, str(IngestionStatus.FAILED))
    # Source should record a failure
    assert source.consecutive_failures == 1
    assert len(run.error_messages) == 1


@pytest.mark.asyncio
async def test_ingest_all_active_topics(fake_async_session, monkeypatch):
    # Patch get_db_session to yield our fake session
    async def fake_get_db_session():
        yield fake_async_session

    import app.services.scheduled_ingestion_service as mod

    monkeypatch.setattr(mod, "get_db_session", lambda: fake_get_db_session())

    # First select in ingest_all_active_topics: active topics list
    t1 = Topic(name="T1", slug="t1", description=None, keywords=[], is_active=True)
    t1.id = uuid4()
    t2 = Topic(name="T2", slug="t2", description=None, keywords=[], is_active=True)
    t2.id = uuid4()
    fake_async_session.execute.return_value = _result_scalars([t1, t2])

    service = ScheduledIngestionService()
    service.ingest_topic = AsyncMock(
        side_effect=[
            IngestionRun(
                topic_id=t1.id,
                status=IngestionStatus.SUCCESS,
                started_at=datetime.now(timezone.utc),
            ),
            IngestionRun(
                topic_id=t2.id,
                status=IngestionStatus.SUCCESS,
                started_at=datetime.now(timezone.utc),
            ),
        ]
    )

    runs = await service.ingest_all_active_topics()
    assert len(runs) == 2
    assert service.ingest_topic.await_count == 2


@pytest.mark.asyncio
async def test_fetch_rss_articles_filters_by_date_threshold(monkeypatch):
    """Test that RSS articles are filtered by the date threshold setting."""
    service = ScheduledIngestionService()

    class FakeResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    async def fake_get(url):
        return FakeResponse(
            "<rss><channel><item><link>https://ex/1</link></item></channel></rss>"
        )

    service.http_client.get = AsyncMock(side_effect=fake_get)

    # Create fake entries with different publication dates
    now = datetime.now(timezone.utc)
    recent_entry = SimpleNamespace(
        link="https://ex/recent", published_parsed=now.timetuple()[:6]  # Recent article
    )
    old_entry = SimpleNamespace(
        link="https://ex/old",
        published_parsed=(now - timedelta(hours=48)).timetuple()[
            :6
        ],  # Old article (48h ago)
    )
    no_date_entry = SimpleNamespace(
        link="https://ex/nodate"
        # No published_parsed attribute
    )

    class FakeFeed:
        entries = [recent_entry, old_entry, no_date_entry]

    import app.services.scheduled_ingestion_service as mod

    monkeypatch.setattr(
        mod, "feedparser", SimpleNamespace(parse=lambda text: FakeFeed())
    )

    # Mock settings with 24h threshold
    with patch.object(mod.settings, "RSS_DATE_THRESHOLD_HOURS", 24):
        src = NewsSource(
            topic_id=uuid4(),
            name="RSS",
            url="https://site/rss.xml",
            source_type=SourceType.RSS,
            config={},
            is_active=True,
        )

        # Act
        urls = await service._fetch_rss_articles(src)

        # Assert - should only include recent article and the one without date
        assert len(urls) == 2
        assert "https://ex/recent" in urls
        assert "https://ex/nodate" in urls
        assert "https://ex/old" not in urls


@pytest.mark.asyncio
async def test_fetch_rss_articles_different_threshold_values(monkeypatch):
    """Test RSS filtering with different threshold values."""
    service = ScheduledIngestionService()

    class FakeResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    async def fake_get(url):
        return FakeResponse("<rss><channel></channel></rss>")

    service.http_client.get = AsyncMock(side_effect=fake_get)

    # Create entries with specific timestamps
    now = datetime.now(timezone.utc)
    test_entries = [
        SimpleNamespace(
            link=f"https://ex/{i}h",
            published_parsed=(now - timedelta(hours=i)).timetuple()[:6],
        )
        for i in [1, 6, 12, 24, 48, 72]  # 1h, 6h, 12h, 24h, 48h, 72h ago
    ]

    class FakeFeed:
        entries = test_entries

    import app.services.scheduled_ingestion_service as mod

    monkeypatch.setattr(
        mod, "feedparser", SimpleNamespace(parse=lambda text: FakeFeed())
    )

    src = NewsSource(
        topic_id=uuid4(),
        name="RSS",
        url="https://site/rss.xml",
        source_type=SourceType.RSS,
        config={},
        is_active=True,
    )

    # Test with 12h threshold - should include 1h and 6h articles
    with patch.object(mod.settings, "RSS_DATE_THRESHOLD_HOURS", 12):
        urls = await service._fetch_rss_articles(src)
        assert len(urls) == 2
        assert "https://ex/1h" in urls
        assert "https://ex/6h" in urls

    # Test with 48h threshold - should include 1h, 6h, 12h, 24h articles
    with patch.object(mod.settings, "RSS_DATE_THRESHOLD_HOURS", 48):
        urls = await service._fetch_rss_articles(src)
        assert len(urls) == 4
        expected_urls = [
            "https://ex/1h",
            "https://ex/6h",
            "https://ex/12h",
            "https://ex/24h",
        ]
        for url in expected_urls:
            assert url in urls


@pytest.mark.asyncio
async def test_fetch_rss_articles_handles_malformed_dates(monkeypatch):
    """Test that RSS fetching handles entries with malformed or missing dates gracefully."""
    service = ScheduledIngestionService()

    class FakeResponse:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            return None

    async def fake_get(url):
        return FakeResponse("<rss><channel></channel></rss>")

    service.http_client.get = AsyncMock(side_effect=fake_get)

    # Create entries with various date scenarios
    valid_entry = SimpleNamespace(
        link="https://ex/valid",
        published_parsed=datetime.now(timezone.utc).timetuple()[:6],
    )
    no_date_entry = SimpleNamespace(link="https://ex/nodate")  # No published_parsed
    empty_date_entry = SimpleNamespace(
        link="https://ex/empty", published_parsed=None  # Explicit None
    )

    class FakeFeed:
        entries = [valid_entry, no_date_entry, empty_date_entry]

    import app.services.scheduled_ingestion_service as mod

    monkeypatch.setattr(
        mod, "feedparser", SimpleNamespace(parse=lambda text: FakeFeed())
    )

    src = NewsSource(
        topic_id=uuid4(),
        name="RSS",
        url="https://site/rss.xml",
        source_type=SourceType.RSS,
        config={},
        is_active=True,
    )

    # Act
    urls = await service._fetch_rss_articles(src)

    # Assert - should include valid entry and entries without dates (fallback behavior)
    assert len(urls) == 3
    assert "https://ex/valid" in urls
    assert "https://ex/nodate" in urls
    assert "https://ex/empty" in urls


def _result_scalar(obj):
    res = MagicMock()
    res.scalar_one_or_none.return_value = obj
    return res


def _result_scalars(items):
    res = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = items
    res.scalars.return_value = scalars
    return res
