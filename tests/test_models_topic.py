from uuid import uuid4

from app.models.topic import (
    ArticleTopic,
    IngestionRun,
    IngestionStatus,
    NewsSource,
    SourceType,
    Topic,
)


def test_topic_column_defaults_and_repr():
    # Validate ORM column defaults without DB
    table = Topic.__table__
    assert table.c.is_active.default.arg is True
    assert table.c.schedule_interval_minutes.default.arg == 60
    assert table.c.created_at.server_default is not None
    assert table.c.updated_at.server_default is not None

    # Create instance and check repr content
    t = Topic(
        name="Tech",
        slug="tech",
        description="All about tech",
        keywords=["hardware", "software"],
    )
    t.id = uuid4()
    t.is_active = True
    assert "Tech" in repr(t)
    assert "active=True" in repr(t)


def test_newssource_column_defaults_and_repr():
    table = NewsSource.__table__
    assert table.c.config.default.arg.__name__ == "dict"
    assert table.c.is_active.default.arg is True
    assert table.c.consecutive_failures.default.arg == 0
    assert table.c.created_at.server_default is not None
    assert table.c.updated_at.server_default is not None

    # Indexes
    index_names = {idx.name for idx in table.indexes}
    assert "ix_news_sources_topic_id" in index_names
    assert "ix_news_sources_source_type" in index_names

    # Instance and repr
    ns = NewsSource(
        topic_id=uuid4(),
        name="Example RSS",
        url="https://example.com/rss.xml",
        source_type=SourceType.RSS,
    )
    ns.id = uuid4()
    assert "Example RSS" in repr(ns)
    assert str(SourceType.RSS) in repr(ns)


def test_ingestionrun_column_defaults_and_repr():
    table = IngestionRun.__table__
    run = IngestionRun(
        topic_id=uuid4(),
        status=IngestionStatus.PENDING,
    )
    run.id = uuid4()
    assert run.status == IngestionStatus.PENDING
    assert "status" in repr(run) or str(IngestionStatus.PENDING) in repr(run)

    assert table.c.articles_discovered.default.arg == 0
    assert table.c.articles_ingested.default.arg == 0
    assert table.c.articles_failed.default.arg == 0
    assert table.c.articles_duplicates.default.arg == 0
    assert table.c.started_at.server_default is not None
    assert table.c.completed_at.server_default is None

    # Indexes
    index_names = {idx.name for idx in table.indexes}
    assert "ix_ingestion_runs_topic_id" in index_names
    assert "ix_ingestion_runs_started_at" in index_names
    assert "ix_ingestion_runs_status" in index_names


def test_article_topic_table_and_defaults():
    table = ArticleTopic.__table__
    # Server default timestamp for created_at
    assert table.c.created_at.server_default is not None

    # Indexes
    index_names = {idx.name for idx in table.indexes}
    assert "ix_article_topics_article_id" in index_names
    assert "ix_article_topics_topic_id" in index_names

    # Create instance (no DB flush)
    at = ArticleTopic(article_id=uuid4(), topic_id=uuid4(), ingestion_run_id=None)
    # created_at is server side, so may be None until persisted
    assert at.article_id is not None
    assert at.topic_id is not None
