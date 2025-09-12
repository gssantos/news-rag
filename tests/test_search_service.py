from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest

from app.services.search_service import SearchService
from tests.conftest import FakeAsyncSession, FakeResult


class TestSearchService:
    """SearchService tests (tests 34-35)"""

    @pytest.mark.asyncio
    async def test_search_calls_embedding_and_maps_rows_to_results(self):
        """Test 34: search calls embedding and maps rows to results"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        fake_llm.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

        service = SearchService(fake_db, fake_llm)

        # Create fake search result rows
        fake_rows = [
            SimpleNamespace(
                id=uuid4(),
                url="https://example.com/article1",
                title="Article 1",
                summary="Summary 1",
                published_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
                score=0.9,
            ),
            SimpleNamespace(
                id=uuid4(),
                url="https://example.com/article2",
                title="Article 2",
                summary="Summary 2",
                published_at=datetime(2023, 1, 2, tzinfo=timezone.utc),
                score=0.8,
            ),
        ]

        # Mock database execute to return fake rows
        fake_result = FakeResult(fake_rows)
        fake_db.execute.return_value = fake_result

        results = await service.search(
            query="test query", k=5, start_date=None, end_date=None
        )

        # Check that embedding was called once
        fake_llm.generate_embedding.assert_called_once_with("test query")

        # Check that db.execute was called once
        fake_db.execute.assert_called_once()

        # Check results
        assert len(results) == 2
        assert results[0].title == "Article 1"
        assert results[0].score == 0.9
        assert results[1].title == "Article 2"
        assert results[1].score == 0.8

    @pytest.mark.asyncio
    async def test_search_applies_date_filters_statement_inspection(self):
        """Test 35: search applies date filters (statement inspection)"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        fake_llm.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

        service = SearchService(fake_db, fake_llm)

        # Mock database to capture the executed statement
        captured_statement = None

        async def capture_execute(stmt):
            nonlocal captured_statement
            captured_statement = stmt
            return FakeResult([])

        fake_db.execute = capture_execute

        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2023, 12, 31, tzinfo=timezone.utc)

        await service.search(
            query="test query", k=5, start_date=start_date, end_date=end_date
        )

        # Check that statement was captured and has date filters
        assert captured_statement is not None

        # Convert to string to check for date filters - this is a simplified check
        # that verifies the date filtering logic is applied
        statement_str = str(captured_statement)

        # Check that date filters are present - the exact SQL syntax may vary
        # but we should see references to published_at filtering
        assert "published_at" in statement_str
        # Check that the statement contains WHERE clauses with date comparisons
        assert "WHERE" in statement_str or "where" in statement_str

    @pytest.mark.asyncio
    async def test_search_no_date_filters_when_none_provided(self):
        """Test that search works without date filters"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        fake_llm.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

        service = SearchService(fake_db, fake_llm)

        fake_result = FakeResult([])
        fake_db.execute.return_value = fake_result

        results = await service.search(
            query="test query", k=5, start_date=None, end_date=None
        )

        # Should execute without error
        assert results == []
        fake_llm.generate_embedding.assert_called_once_with("test query")

    @pytest.mark.asyncio
    async def test_search_with_only_start_date(self):
        """Test search with only start date filter"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        fake_llm.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

        service = SearchService(fake_db, fake_llm)
        fake_result = FakeResult([])
        fake_db.execute.return_value = fake_result

        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)

        results = await service.search(
            query="test query", k=5, start_date=start_date, end_date=None
        )

        assert results == []
        fake_llm.generate_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_only_end_date(self):
        """Test search with only end date filter"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        fake_llm.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])

        service = SearchService(fake_db, fake_llm)
        fake_result = FakeResult([])
        fake_db.execute.return_value = fake_result

        end_date = datetime(2023, 12, 31, tzinfo=timezone.utc)

        results = await service.search(
            query="test query", k=5, start_date=None, end_date=end_date
        )

        assert results == []
        fake_llm.generate_embedding.assert_called_once()
