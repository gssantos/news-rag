from unittest.mock import AsyncMock, Mock, patch

import httpx
import pytest
from fastapi import HTTPException

from app.core.security import (
    get_api_key,
    validate_redirect_target_security,
    validate_url_security,
)


class TestSecurity:
    """Security tests (tests 5-12)"""

    def test_get_api_key_returns_none_when_disabled(self):
        """Test 5: get_api_key returns None when disabled"""
        with patch("app.core.security.settings") as mock_settings:
            mock_settings.API_KEY = None
            result = get_api_key(None)
            assert result is None

    def test_get_api_key_requires_exact_match(self):
        """Test 6: get_api_key requires exact match"""
        with patch("app.core.security.settings") as mock_settings:
            mock_settings.API_KEY = "secret"

            # Correct key should work
            result = get_api_key("secret")
            assert result == "secret"

            # Wrong key should raise exception
            with pytest.raises(HTTPException) as exc_info:
                get_api_key("wrong")
            assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_validate_url_security_rejects_invalid_scheme(self):
        """Test 7: validate_url_security rejects invalid scheme"""
        with pytest.raises(HTTPException) as exc_info:
            await validate_url_security("ftp://example.com")
        assert exc_info.value.status_code == 400
        assert "Invalid scheme" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_validate_url_security_rejects_domain_not_in_allowlist(self):
        """Test 8: validate_url_security rejects domain not in allowlist"""
        with patch("app.core.security.settings") as mock_settings:
            mock_settings.ALLOWED_DOMAINS = ["example.com"]

            with pytest.raises(HTTPException) as exc_info:
                await validate_url_security("https://not-example.com")
            assert exc_info.value.status_code == 400
            assert "not in the allowlist" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_validate_url_security_blocks_private_ip_resolution(self):
        """Test 9: validate_url_security blocks private IP resolution"""
        with patch("app.core.security.settings") as mock_settings:
            mock_settings.ALLOWED_DOMAINS = []

            mock_getaddrinfo = AsyncMock(
                return_value=[("", "", "", "", ("10.0.0.1", 0))]
            )

            with patch("asyncio.get_running_loop") as mock_get_loop:
                mock_loop = Mock()
                mock_loop.getaddrinfo = mock_getaddrinfo
                mock_get_loop.return_value = mock_loop

                with pytest.raises(HTTPException) as exc_info:
                    await validate_url_security("https://blocked.example.com")
                assert exc_info.value.status_code == 400
                assert "private, loopback, or reserved IP ranges" in str(
                    exc_info.value.detail
                )

    @pytest.mark.asyncio
    async def test_validate_url_security_allows_public_ip_resolution(self):
        """Test 10: validate_url_security allows public IP resolution"""
        with patch("app.core.security.settings") as mock_settings:
            mock_settings.ALLOWED_DOMAINS = []

            mock_getaddrinfo = AsyncMock(
                return_value=[
                    ("", "", "", "", ("93.184.216.34", 0))  # example.com public IP
                ]
            )

            with patch("asyncio.get_running_loop") as mock_get_loop:
                mock_loop = Mock()
                mock_loop.getaddrinfo = mock_getaddrinfo
                mock_get_loop.return_value = mock_loop

                result = await validate_url_security("https://example.com")
                assert result is True

    @pytest.mark.asyncio
    async def test_validate_redirect_target_security_revalidates_redirect(self):
        """Test 11: validate_redirect_target_security revalidates redirect"""
        original_url = "https://good.example.com"
        redirect_url = "https://evil.local"

        # Mock the HEAD request to return a redirect
        mock_response = Mock()
        mock_response.url = redirect_url

        mock_client = AsyncMock()
        mock_client.head.return_value = mock_response
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("httpx.AsyncClient", return_value=mock_client):
            with patch("app.core.security.validate_url_security") as mock_validate:
                # First call succeeds, second call (for redirect URL) raises an exception
                def validate_side_effect(url):
                    if url == redirect_url:
                        raise HTTPException(status_code=400, detail="Blocked")
                    return True

                mock_validate.side_effect = validate_side_effect

                with pytest.raises(HTTPException):
                    await validate_redirect_target_security(original_url)

    @pytest.mark.asyncio
    async def test_validate_redirect_target_security_ignores_head_failure(self):
        """Test 12: validate_redirect_target_security ignores HEAD failure"""
        mock_client = AsyncMock()
        mock_client.head.side_effect = httpx.TimeoutException("Timeout")
        mock_client.__aenter__.return_value = mock_client
        mock_client.__aexit__.return_value = None

        with patch("httpx.AsyncClient", return_value=mock_client):
            # Should not raise exception, returns True
            result = await validate_redirect_target_security("https://example.com")
            assert result is True
