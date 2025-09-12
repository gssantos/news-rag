import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastapi import HTTPException
from langchain_core.documents import Document

from app.schemas.article import ArticleIngestRequest
from app.services.ingestion_service import IngestionService
from tests.conftest import FakeAsyncSession, create_fake_article


class TestIngestionService:
    """IngestionService tests (tests 21-33)"""

    @pytest.mark.asyncio
    async def test_ingest_url_returns_existing_when_found_and_force_false(self):
        """Test 21: ingest_url returns existing when found and force=False"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        service = IngestionService(fake_db, fake_llm)

        # Create fake existing article
        existing_article = create_fake_article()
        service._find_article_by_url = AsyncMock(return_value=existing_article)

        with patch(
            "app.services.ingestion_service.validate_url_security"
        ) as mock_validate:
            mock_validate.return_value = True

            request = ArticleIngestRequest(url="https://example.com", force=False)
            result_article, status = await service.ingest_url(request)

            assert result_article == existing_article
            assert status == "existing"
            # LLM methods should not be called
            assert not fake_llm.generate_summary.called
            assert not fake_llm.generate_embedding.called

    @pytest.mark.asyncio
    async def test_ingest_url_processes_when_force_true_even_if_exists(self):
        """Test 22: ingest_url processes when force=True even if exists"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        fake_llm.generate_summary = AsyncMock(return_value="Test summary")
        fake_llm.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        fake_llm.compose_embedding_input = Mock(return_value="combined text")
        fake_llm.llm_model_name = "gpt-5-mini"
        fake_llm.emb_model_name = "text-embedding-3-small"
        fake_llm.emb_dim = 1536

        service = IngestionService(fake_db, fake_llm)

        # Mock existing article found
        existing_article = create_fake_article()
        service._find_article_by_url = AsyncMock(return_value=existing_article)

        # Mock extraction
        fake_doc = Document(page_content="Article content")
        fake_metadata = {
            "title": "Test Title",
            "source_domain": "example.com",
            "published_at": datetime(2023, 1, 1, tzinfo=timezone.utc),
        }
        service._fetch_and_extract = AsyncMock(return_value=(fake_doc, fake_metadata))

        # Mock upsert
        updated_article = create_fake_article()
        service._upsert_article = AsyncMock(return_value=(updated_article, False))

        with patch("app.services.ingestion_service.validate_url_security"):
            request = ArticleIngestRequest(url="https://example.com", force=True)
            result_article, status = await service.ingest_url(request)

            assert result_article == updated_article
            assert status == "updated"
            # Both LLM methods should be called
            fake_llm.generate_summary.assert_called_once()
            fake_llm.generate_embedding.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_and_extract_uses_newsurl_loader_successfully(self):
        """Test 23: _fetch_and_extract uses NewsURLLoader successfully"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        service = IngestionService(fake_db, fake_llm)

        # Mock successful NewsURLLoader
        fake_doc = Document(
            page_content="Article content",
            metadata={"title": "Test Title", "publish_date": "2020-01-02T03:04:05Z"},
        )

        mock_loader = Mock()
        mock_loader.load.return_value = [fake_doc]

        with patch(
            "app.services.ingestion_service.NewsURLLoader", return_value=mock_loader
        ):
            with patch("asyncio.to_thread", return_value=[fake_doc]):
                with patch("asyncio.wait_for", return_value=[fake_doc]):
                    result_doc, metadata = await service._fetch_and_extract(
                        "https://example.com", None
                    )

                    assert result_doc == fake_doc
                    assert metadata["title"] == "Test Title"
                    assert metadata["source_domain"] == "example.com"
                    assert metadata["published_at"].year == 2020

    @pytest.mark.asyncio
    async def test_fetch_and_extract_falls_back_to_webbase_loader(self):
        """Test 24: _fetch_and_extract falls back to WebBaseLoader when NewsURLLoader fails"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        service = IngestionService(fake_db, fake_llm)

        # Mock successful WebBaseLoader
        fake_doc = Document(
            page_content="Article content", metadata={"title": "Test Title"}
        )
        mock_web_loader = Mock()
        mock_web_loader.load.return_value = [fake_doc]

        with patch("app.services.ingestion_service.NewsURLLoader"):
            with patch(
                "app.services.ingestion_service.WebBaseLoader",
                return_value=mock_web_loader,
            ):
                with patch("asyncio.to_thread") as mock_to_thread:
                    # First call (NewsURLLoader) fails, second call (WebBaseLoader) succeeds
                    mock_to_thread.side_effect = [
                        ValueError("NewsURL failed"),
                        [fake_doc],
                    ]
                    with patch("asyncio.wait_for") as mock_wait:
                        # First call throws exception, second call returns result
                        mock_wait.side_effect = [
                            ValueError("NewsURL failed"),
                            [fake_doc],
                        ]

                        result_doc, metadata = await service._fetch_and_extract(
                            "https://example.com", None
                        )

                        assert result_doc == fake_doc
                        assert metadata["source_domain"] == "example.com"

    @pytest.mark.asyncio
    async def test_fetch_and_extract_returns_504_when_both_loaders_timeout(self):
        """Test 25: _fetch_and_extract returns 504 when both loaders time out"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        service = IngestionService(fake_db, fake_llm)

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with pytest.raises(HTTPException) as exc_info:
                await service._fetch_and_extract("https://example.com", None)

            assert exc_info.value.status_code == 504
            assert "Timeout" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_fetch_and_extract_raises_422_when_both_loaders_fail_generically(
        self,
    ):
        """Test 26: _fetch_and_extract raises 422 when both loaders fail generically"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        service = IngestionService(fake_db, fake_llm)

        # Mock both loaders to fail generically
        with patch(
            "asyncio.to_thread",
            side_effect=[ValueError("Primary failed"), Exception("Fallback failed")],
        ):
            with patch("asyncio.wait_for", side_effect=lambda coro, timeout: coro):
                # This should be caught at the ingest_url level and converted to 422
                service._find_article_by_url = AsyncMock(return_value=None)

                with patch("app.services.ingestion_service.validate_url_security"):
                    request = ArticleIngestRequest(url="https://example.com")
                    with pytest.raises(HTTPException) as exc_info:
                        await service.ingest_url(request)

                    assert exc_info.value.status_code == 422
                    assert "Failed to extract content" in str(exc_info.value.detail)

    def test_extract_metadata_parses_string_date_and_makes_timezone_aware(self):
        """Test 27: _extract_metadata parses string date and makes it timezone-aware"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        service = IngestionService(fake_db, fake_llm)

        loader_metadata = {
            "publish_date": "2020-01-02 03:04:05"
        }  # naive datetime string

        result = service._extract_metadata(loader_metadata, "https://example.com", None)

        assert result["published_at"] is not None
        assert result["published_at"].tzinfo is not None  # timezone-aware
        assert result["published_at"].tzinfo == timezone.utc

    def test_extract_metadata_respects_override_published_at(self):
        """Test 28: _extract_metadata respects override_published_at"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        service = IngestionService(fake_db, fake_llm)

        override_date = datetime(2023, 5, 1, tzinfo=timezone.utc)
        loader_metadata = {"publish_date": "2020-01-02 03:04:05"}

        result = service._extract_metadata(
            loader_metadata, "https://example.com", override_date
        )

        assert result["published_at"] == override_date

    @pytest.mark.asyncio
    async def test_ingest_url_persists_via_upsert_article_and_returns_created(self):
        """Test 29: ingest_url persists via _upsert_article and returns created"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        fake_llm.generate_summary = AsyncMock(return_value="Test summary")
        fake_llm.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        fake_llm.compose_embedding_input = Mock(return_value="combined text")
        fake_llm.llm_model_name = "gpt-5-mini"
        fake_llm.emb_model_name = "text-embedding-3-small"
        fake_llm.emb_dim = 1536

        service = IngestionService(fake_db, fake_llm)

        # Mock no existing article
        service._find_article_by_url = AsyncMock(return_value=None)

        # Mock extraction
        fake_doc = Document(page_content="Article content")
        fake_metadata = {
            "title": "Test Title",
            "source_domain": "example.com",
            "published_at": datetime(2023, 1, 1, tzinfo=timezone.utc),
        }
        service._fetch_and_extract = AsyncMock(return_value=(fake_doc, fake_metadata))

        # Mock upsert returning new article
        new_article = create_fake_article()
        service._upsert_article = AsyncMock(return_value=(new_article, True))

        with patch("app.services.ingestion_service.validate_url_security"):
            request = ArticleIngestRequest(url="https://example.com")
            result_article, status = await service.ingest_url(request)

            assert result_article == new_article
            assert status == "created"

            # Check that _upsert_article was called with correct data
            call_args = service._upsert_article.call_args[0][0]
            assert call_args["llm_model"] == fake_llm.llm_model_name
            assert call_args["embed_model"] == fake_llm.emb_model_name
            assert call_args["embed_dim"] == fake_llm.emb_dim

    @pytest.mark.asyncio
    async def test_ingest_url_persists_via_upsert_article_and_returns_updated(self):
        """Test 30: ingest_url persists via _upsert_article and returns updated"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        fake_llm.generate_summary = AsyncMock(return_value="Test summary")
        fake_llm.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        fake_llm.compose_embedding_input = Mock(return_value="combined text")
        fake_llm.llm_model_name = "gpt-5-mini"
        fake_llm.emb_model_name = "text-embedding-3-small"
        fake_llm.emb_dim = 1536

        service = IngestionService(fake_db, fake_llm)

        # Mock no existing article (or force=True scenario)
        service._find_article_by_url = AsyncMock(return_value=None)

        # Mock extraction
        fake_doc = Document(page_content="Article content")
        fake_metadata = {
            "title": "Test Title",
            "source_domain": "example.com",
            "published_at": datetime(2023, 1, 1, tzinfo=timezone.utc),
        }
        service._fetch_and_extract = AsyncMock(return_value=(fake_doc, fake_metadata))

        # Mock upsert returning updated article
        updated_article = create_fake_article()
        service._upsert_article = AsyncMock(return_value=(updated_article, False))

        with patch("app.services.ingestion_service.validate_url_security"):
            request = ArticleIngestRequest(url="https://example.com")
            result_article, status = await service.ingest_url(request)

            assert result_article == updated_article
            assert status == "updated"

    @pytest.mark.asyncio
    async def test_ingest_url_propagates_llm_summary_error_as_502(self):
        """Test 31: ingest_url propagates LLM summary error as 502"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        fake_llm.generate_summary = AsyncMock(
            side_effect=HTTPException(status_code=502, detail="LLM Error")
        )

        service = IngestionService(fake_db, fake_llm)

        # Mock no existing article
        service._find_article_by_url = AsyncMock(return_value=None)

        # Mock extraction
        fake_doc = Document(page_content="Article content")
        fake_metadata = {
            "title": "Test Title",
            "source_domain": "example.com",
            "published_at": None,
        }
        service._fetch_and_extract = AsyncMock(return_value=(fake_doc, fake_metadata))

        with patch("app.services.ingestion_service.validate_url_security"):
            request = ArticleIngestRequest(url="https://example.com")
            with pytest.raises(HTTPException) as exc_info:
                await service.ingest_url(request)

            assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_ingest_url_propagates_embedding_error_as_502(self):
        """Test 32: ingest_url propagates embedding error as 502"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        fake_llm.generate_summary = AsyncMock(return_value="Test summary")
        fake_llm.generate_embedding = AsyncMock(
            side_effect=HTTPException(status_code=502, detail="Embedding Error")
        )
        fake_llm.compose_embedding_input = Mock(return_value="combined text")

        service = IngestionService(fake_db, fake_llm)

        # Mock no existing article
        service._find_article_by_url = AsyncMock(return_value=None)

        # Mock extraction
        fake_doc = Document(page_content="Article content")
        fake_metadata = {
            "title": "Test Title",
            "source_domain": "example.com",
            "published_at": None,
        }
        service._fetch_and_extract = AsyncMock(return_value=(fake_doc, fake_metadata))

        with patch("app.services.ingestion_service.validate_url_security"):
            request = ArticleIngestRequest(url="https://example.com")
            with pytest.raises(HTTPException) as exc_info:
                await service.ingest_url(request)

            assert exc_info.value.status_code == 502

    @pytest.mark.asyncio
    async def test_ingest_url_returns_500_on_db_upsert_failure(self):
        """Test 33: ingest_url returns 500 on DB upsert failure"""
        fake_db = FakeAsyncSession()
        fake_llm = Mock()
        fake_llm.generate_summary = AsyncMock(return_value="Test summary")
        fake_llm.generate_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
        fake_llm.compose_embedding_input = Mock(return_value="combined text")

        service = IngestionService(fake_db, fake_llm)

        # Mock no existing article
        service._find_article_by_url = AsyncMock(return_value=None)

        # Mock extraction
        fake_doc = Document(page_content="Article content")
        fake_metadata = {
            "title": "Test Title",
            "source_domain": "example.com",
            "published_at": None,
        }
        service._fetch_and_extract = AsyncMock(return_value=(fake_doc, fake_metadata))

        # Mock upsert to raise HTTPException 500
        service._upsert_article = AsyncMock(
            side_effect=HTTPException(status_code=500, detail="DB Error")
        )

        with patch("app.services.ingestion_service.validate_url_security"):
            request = ArticleIngestRequest(url="https://example.com")
            with pytest.raises(HTTPException) as exc_info:
                await service.ingest_url(request)

            assert exc_info.value.status_code == 500
