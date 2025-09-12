from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from app.core.scheduler import IngestionScheduler
from app.models.topic import Topic


class FakeAsyncSessionCtx:
    """Async context manager returning a provided fake async DB session."""

    def __init__(self, session):
        self._session = session

    async def __aenter__(self):
        return self._session

    async def __aexit__(self, exc_type, exc, tb):
        return False


@pytest.fixture
def fake_apscheduler(monkeypatch):
    """Patch AsyncIOScheduler inside IngestionScheduler to a MagicMock instance."""
    scheduler_mock = MagicMock()
    scheduler_mock.get_job.return_value = None
    scheduler_mock.add_job = MagicMock()
    scheduler_mock.remove_job = MagicMock()
    scheduler_mock.remove_all_jobs = MagicMock()
    scheduler_mock.start = MagicMock()
    scheduler_mock.shutdown = MagicMock()

    class FakeFactory:
        def __call__(self, *args, **kwargs):
            return scheduler_mock

    import app.core.scheduler as mod

    monkeypatch.setattr(mod, "AsyncIOScheduler", FakeFactory())
    return scheduler_mock


@pytest.mark.asyncio
async def test_scheduler_start_initializes_and_starts(fake_apscheduler, monkeypatch):
    sched = IngestionScheduler()
    # Avoid DB work in reload_topic_schedules
    sched.reload_topic_schedules = AsyncMock()

    await sched.start()

    assert sched.ingestion_service is not None
    sched.reload_topic_schedules.assert_awaited()
    fake_apscheduler.start.assert_called_once()


@pytest.mark.asyncio
async def test_reload_topic_schedules_removes_and_schedules(
    monkeypatch, fake_apscheduler
):
    # Fake session factory
    fake_session = MagicMock()
    fake_session.close = AsyncMock()  # Make close async
    fake_session.execute = AsyncMock()  # Make execute async

    t1 = Topic(
        name="T1",
        slug="t1",
        description=None,
        keywords=[],
        is_active=True,
        schedule_interval_minutes=60,
    )
    t1.id = uuid4()
    t2 = Topic(
        name="T2",
        slug="t2",
        description=None,
        keywords=[],
        is_active=True,
        schedule_interval_minutes=60,
    )
    t2.id = uuid4()
    # First execute returns active topics
    res = MagicMock()
    scalars = MagicMock()
    scalars.all.return_value = [t1, t2]
    res.scalars.return_value = scalars
    fake_session.execute.return_value = res

    import app.core.scheduler as mod

    monkeypatch.setattr(
        mod, "AsyncSessionLocal", lambda: FakeAsyncSessionCtx(fake_session)
    )

    sched = IngestionScheduler()
    # Spy on schedule_topic
    sched.schedule_topic = MagicMock()

    await sched.reload_topic_schedules()

    fake_apscheduler.remove_all_jobs.assert_called_once()
    assert sched.schedule_topic.call_count == 2


def test_schedule_topic_adds_job(fake_apscheduler):
    sched = IngestionScheduler()

    topic = Topic(
        name="AI",
        slug="ai",
        description=None,
        keywords=["ai"],
        is_active=True,
        schedule_interval_minutes=10,
    )
    topic.id = uuid4()
    topic.name = "AI"

    sched.schedule_topic(topic)
    # Ensure add_job called with correct args
    assert fake_apscheduler.add_job.called
    call = fake_apscheduler.add_job.call_args
    kwargs = call.kwargs
    assert kwargs["args"] == [str(topic.id)]
    assert kwargs["id"] == f"ingest_topic_{topic.id}"
    assert "Ingest articles for topic" in kwargs["name"]


def test_schedule_topic_replaces_existing_job(fake_apscheduler):
    sched = IngestionScheduler()
    topic = Topic(
        name="AI",
        slug="ai",
        description=None,
        keywords=[],
        is_active=True,
        schedule_interval_minutes=60,
    )
    topic.id = uuid4()

    # Simulate existing job
    fake_apscheduler.get_job.return_value = MagicMock()
    sched.schedule_topic(topic)
    fake_apscheduler.remove_job.assert_called_with(f"ingest_topic_{topic.id}")


@pytest.mark.asyncio
async def test_run_topic_ingestion_invokes_service(monkeypatch, fake_apscheduler):
    sched = IngestionScheduler()
    # Fake DB session factory
    fake_session = MagicMock()
    fake_session.close = AsyncMock()  # Make close async

    import app.core.scheduler as mod

    monkeypatch.setattr(
        mod, "AsyncSessionLocal", lambda: FakeAsyncSessionCtx(fake_session)
    )

    # Fake ingestion service context
    class FakeService:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        async def ingest_topic(self, topic_id, db):
            return None

    sched.ingestion_service = FakeService()
    # Spy
    sched.ingestion_service.ingest_topic = AsyncMock(return_value=None)

    topic_id = str(uuid4())
    await sched._run_topic_ingestion(topic_id)
    sched.ingestion_service.ingest_topic.assert_awaited()
    args, kwargs = sched.ingestion_service.ingest_topic.call_args
    assert args[0] == topic_id
    assert args[1] == fake_session


def test_update_pause_resume_remove_topic(fake_apscheduler):
    sched = IngestionScheduler()
    topic_id = str(uuid4())

    # Update schedule when job exists
    job = MagicMock()
    fake_apscheduler.get_job.return_value = job
    sched.update_topic_schedule(topic_id, 25)
    assert job.reschedule.called

    # Pause/resume when job exists
    sched.pause_topic(topic_id)
    job.pause.assert_called_once()

    sched.resume_topic(topic_id)
    job.resume.assert_called_once()

    # Remove when job exists
    sched.remove_topic(topic_id)
    fake_apscheduler.remove_job.assert_called_with(f"ingest_topic_{topic_id}")

    # No-op when no job
    fake_apscheduler.get_job.return_value = None
    sched.update_topic_schedule(topic_id, 10)  # Should not raise
    sched.pause_topic(topic_id)  # Should not raise
    sched.resume_topic(topic_id)  # Should not raise
    sched.remove_topic(topic_id)  # Should not raise


@pytest.mark.asyncio
async def test_shutdown_closes_scheduler_and_service(fake_apscheduler):
    sched = IngestionScheduler()

    # Inject a fake ingestion service with __aexit__
    class FakeService:
        async def __aexit__(self, exc_type, exc, tb):
            return False

    fake_service = FakeService()
    fake_service.__aexit__ = AsyncMock()
    sched.ingestion_service = fake_service

    await sched.shutdown()
    fake_apscheduler.shutdown.assert_called_once()
    fake_service.__aexit__.assert_awaited()
