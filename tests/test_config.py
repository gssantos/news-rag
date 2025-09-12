import logging
from unittest.mock import patch

import pytest

from app.core.config import Settings


class TestSettings:
    """Core config tests (tests 1-4)"""

    def test_validates_asyncpg_driver(self):
        """Test 1: Settings validates asyncpg driver"""
        with pytest.raises(
            ValueError, match="DATABASE_URL must use the 'postgresql\\+asyncpg' driver"
        ):
            settings = Settings(
                DATABASE_URL="postgresql://user:pass@host/db",  # non-async driver
                OPENAI_API_KEY="test-key",
            )
            settings.validate_config()

    def test_allowed_domains_parsing(self):
        """Test 2: ALLOWED_DOMAINS parsing"""
        with patch.dict(
            "os.environ",
            {
                "DATABASE_URL": "postgresql+asyncpg://user:pass@host/db",
                "ALLOWED_DOMAINS": "example.com, sub.test.org",
                "OPENAI_API_KEY": "test-key",
            },
        ):
            settings = Settings()
            assert settings.ALLOWED_DOMAINS == ["example.com", "sub.test.org"]

    def test_openai_warning_when_key_missing(self, caplog):
        """Test 3: OpenAI warning when key missing"""
        with caplog.at_level(logging.WARNING):
            settings = Settings(
                DATABASE_URL="postgresql+asyncpg://user:pass@host/db",
                LLM_PROVIDER="openai",
                EMB_PROVIDER="openai",
                OPENAI_API_KEY=None,
            )
            settings.validate_config()

        assert any(
            "OPENAI_API_KEY is not set" in record.message for record in caplog.records
        )

    def test_get_log_level_returns_mapped_level(self):
        """Test 4: get_log_level returns mapped level"""
        settings = Settings(
            DATABASE_URL="postgresql+asyncpg://user:pass@host/db",
            LOG_LEVEL="DEBUG",
            OPENAI_API_KEY="test-key",
        )
        assert settings.get_log_level() == logging.DEBUG

    def test_allowed_domains_empty_when_none(self):
        """Test that ALLOWED_DOMAINS returns empty list when not set"""
        settings = Settings(
            DATABASE_URL="postgresql+asyncpg://user:pass@host/db",
            ALLOWED_DOMAINS_STR=None,
            OPENAI_API_KEY="test-key",
        )
        assert settings.ALLOWED_DOMAINS == []
