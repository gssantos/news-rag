import pytest
from pydantic import HttpUrl, ValidationError

from app.schemas.article import ArticleBase, ArticleIngestRequest


class TestSchemas:
    """Schema tests (tests 13-14)"""

    def test_article_ingest_request_accepts_http_https_only(self):
        """Test 13: ArticleIngestRequest accepts http/https only"""
        # Valid URLs should work
        valid_request = ArticleIngestRequest(url="http://example.com")
        assert str(valid_request.url) == "http://example.com/"

        valid_request_https = ArticleIngestRequest(url="https://example.com")
        assert str(valid_request_https.url) == "https://example.com/"

        # Invalid scheme should raise validation error
        with pytest.raises(ValidationError) as exc_info:
            ArticleIngestRequest(url="ftp://example.com")

        # Check that the error is related to URL validation
        assert "url" in str(exc_info.value)

    def test_response_url_is_always_string(self):
        """Test 14: Response url is always string"""
        from datetime import datetime, timezone
        from uuid import uuid4

        # Test with string input
        article_data = {
            "id": uuid4(),
            "url": "https://example.com",
            "title": "Test",
            "published_at": datetime(2023, 1, 1, tzinfo=timezone.utc),
            "created_at": datetime(2023, 1, 1, tzinfo=timezone.utc),
            "updated_at": datetime(2023, 1, 1, tzinfo=timezone.utc),
        }

        article = ArticleBase(**article_data)
        assert isinstance(article.url, str)
        assert article.url == "https://example.com"

        # Test with HttpUrl input (simulating what might come from Pydantic)
        article_data_with_httpurl = article_data.copy()
        article_data_with_httpurl["url"] = HttpUrl("https://example.com")

        article_from_httpurl = ArticleBase(**article_data_with_httpurl)
        assert isinstance(article_from_httpurl.url, str)
        assert article_from_httpurl.url == "https://example.com/"
