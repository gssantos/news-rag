"""Unit tests for topic API endpoints."""

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

from app.models.topic import IngestionStatus


class TestTopicAPI:
    """Test suite for topic management API endpoints."""

    def test_create_topic_success(
        self, api_client, fake_async_session, mock_scheduler_endpoints
    ):
        """Test creating a new topic successfully."""
        # Mock that no existing topic is found
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        fake_async_session.execute.return_value = result_mock

        # Mock refresh to set ID and timestamps
        def refresh_side_effect(obj):
            if not hasattr(obj, "id") or obj.id is None:
                obj.id = uuid4()
            now = datetime.now(timezone.utc)
            obj.created_at = now
            obj.updated_at = now

        fake_async_session.refresh.side_effect = refresh_side_effect

        payload = {
            "name": "Energy Markets",
            "slug": "energy-markets",
            "description": "Energy market news",
            "keywords": ["energy", "oil", "gas"],
            "is_active": True,
            "schedule_interval_minutes": 60,
        }

        response = api_client.post("/api/v1/topics", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "Energy Markets"
        assert data["slug"] == "energy-markets"
        assert data["keywords"] == ["energy", "oil", "gas"]
        assert "id" in data

        # Verify scheduler was called for active topic
        mock_scheduler_endpoints.schedule_topic.assert_called_once()

    def test_create_topic_conflict(self, api_client, fake_async_session, topic_factory):
        """Test creating a topic with existing name/slug returns conflict."""
        # Mock that an existing topic is found
        existing_topic = topic_factory(name="Existing", slug="existing")
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = existing_topic
        fake_async_session.execute.return_value = result_mock

        payload = {"name": "Existing", "slug": "existing", "keywords": []}

        response = api_client.post("/api/v1/topics", json=payload)

        assert response.status_code == 409
        assert "already exists" in response.json()["message"]

    def test_list_topics(self, api_client, fake_async_session, topic_factory):
        """Test listing all topics."""
        topics = [
            topic_factory(name="Topic 1", slug="topic-1"),
            topic_factory(name="Topic 2", slug="topic-2"),
        ]

        result_mock = MagicMock()
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = topics
        result_mock.scalars.return_value = scalars_mock
        fake_async_session.execute.return_value = result_mock

        response = api_client.get("/api/v1/topics")

        assert response.status_code == 200
        data = response.json()
        assert "topics" in data
        assert len(data["topics"]) == 2

    def test_get_topic_detail(
        self, api_client, fake_async_session, topic_factory, source_factory
    ):
        """Test getting topic details including sources."""
        topic = topic_factory(name="AI News", slug="ai-news")
        source = source_factory(topic_id=topic.id, name="AI RSS Feed")
        topic.sources = [source]

        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = topic
        fake_async_session.execute.return_value = result_mock

        response = api_client.get(f"/api/v1/topics/{topic.id}")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "AI News"
        assert len(data["sources"]) == 1
        assert data["sources"][0]["name"] == "AI RSS Feed"

    def test_get_topic_not_found(self, api_client, fake_async_session):
        """Test getting non-existent topic returns 404."""
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = None
        fake_async_session.execute.return_value = result_mock

        response = api_client.get(f"/api/v1/topics/{uuid4()}")

        assert response.status_code == 404

    def test_update_topic(
        self, api_client, fake_async_session, topic_factory, mock_scheduler_endpoints
    ):
        """Test updating a topic."""
        topic = topic_factory(name="Old Name", is_active=True)
        fake_async_session.get.return_value = topic

        # Mock refresh
        def refresh_side_effect(obj):
            now = datetime.now(timezone.utc)
            obj.updated_at = now

        fake_async_session.refresh.side_effect = refresh_side_effect

        payload = {"name": "New Name", "is_active": False}

        response = api_client.patch(f"/api/v1/topics/{topic.id}", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "New Name"
        assert data["is_active"] is False

        # Verify scheduler was updated (removed for inactive)
        mock_scheduler_endpoints.remove_topic.assert_called_once()

    def test_delete_topic(
        self, api_client, fake_async_session, topic_factory, mock_scheduler_endpoints
    ):
        """Test deleting a topic."""
        topic = topic_factory()
        fake_async_session.get.return_value = topic

        response = api_client.delete(f"/api/v1/topics/{topic.id}")

        assert response.status_code == 204

        # Verify scheduler removal and DB delete
        mock_scheduler_endpoints.remove_topic.assert_called_once()
        fake_async_session.delete.assert_awaited()

    def test_add_source_to_topic(self, api_client, fake_async_session, topic_factory):
        """Test adding a source to a topic."""
        topic = topic_factory()
        fake_async_session.get.return_value = topic

        # Mock refresh for the new source
        def refresh_side_effect(obj):
            if not hasattr(obj, "id") or obj.id is None:
                obj.id = uuid4()
            now = datetime.now(timezone.utc)
            obj.created_at = now
            obj.updated_at = now
            # Set consecutive_failures to 0 if not already set
            if (
                not hasattr(obj, "consecutive_failures")
                or obj.consecutive_failures is None
            ):
                obj.consecutive_failures = 0

        fake_async_session.refresh.side_effect = refresh_side_effect

        payload = {
            "name": "News RSS Feed",
            "url": "https://example.com/rss.xml",
            "source_type": "rss",
            "is_active": True,
        }

        response = api_client.post(f"/api/v1/topics/{topic.id}/sources", json=payload)

        assert response.status_code == 201
        data = response.json()
        assert data["name"] == "News RSS Feed"
        assert data["url"] == "https://example.com/rss.xml"
        assert data["source_type"] == "rss"

    def test_list_sources(self, api_client, fake_async_session, source_factory):
        """Test listing sources for a topic."""
        sources = [source_factory(name="Source 1"), source_factory(name="Source 2")]

        result_mock = MagicMock()
        scalars_mock = MagicMock()
        scalars_mock.all.return_value = sources
        result_mock.scalars.return_value = scalars_mock
        fake_async_session.execute.return_value = result_mock

        response = api_client.get(f"/api/v1/topics/{uuid4()}/sources")

        assert response.status_code == 200
        data = response.json()
        assert len(data) == 2
        assert data[0]["name"] == "Source 1"

    def test_delete_source(self, api_client, fake_async_session, source_factory):
        """Test deleting a source from a topic."""
        source = source_factory()
        result_mock = MagicMock()
        result_mock.scalar_one_or_none.return_value = source
        fake_async_session.execute.return_value = result_mock

        response = api_client.delete(
            f"/api/v1/topics/{source.topic_id}/sources/{source.id}"
        )

        assert response.status_code == 204
        fake_async_session.delete.assert_awaited()

    def test_get_ingestion_stats(
        self, api_client, fake_async_session, topic_factory, ingestion_run_factory
    ):
        """Test getting ingestion statistics."""
        # Mock active topics query
        active_topics = [topic_factory(is_active=True)]
        active_result = MagicMock()
        active_scalars = MagicMock()
        active_scalars.all.return_value = active_topics
        active_result.scalars.return_value = active_scalars

        # Mock total topics query
        total_topics = [topic_factory(), topic_factory()]
        total_result = MagicMock()
        total_scalars = MagicMock()
        total_scalars.all.return_value = total_topics
        total_result.scalars.return_value = total_scalars

        # Mock recent runs query
        runs = [
            ingestion_run_factory(
                status=IngestionStatus.SUCCESS, articles_ingested=5, articles_failed=1
            ),
            ingestion_run_factory(
                status=IngestionStatus.SUCCESS, articles_ingested=3, articles_failed=0
            ),
        ]
        runs_result = MagicMock()
        runs_scalars = MagicMock()
        runs_scalars.all.return_value = runs
        runs_result.scalars.return_value = runs_scalars

        # Set up execute side effects
        fake_async_session.execute.side_effect = [
            active_result,
            total_result,
            runs_result,
        ]

        response = api_client.get("/api/v1/topics/stats/summary")

        assert response.status_code == 200
        data = response.json()
        assert data["active_topics"] == 1
        assert data["total_topics"] == 2
        assert data["total_articles_ingested_recently"] == 8
        assert data["total_articles_failed_recently"] == 1
