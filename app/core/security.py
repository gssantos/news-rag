import asyncio
import ipaddress
import logging
import socket
from typing import List
from urllib.parse import urlparse

import httpx
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

from app.core.config import settings

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def get_api_key(api_key: str = Security(api_key_header)):
    """Dependency to enforce API Key authentication if configured."""
    if settings.API_KEY:
        if api_key == settings.API_KEY:
            return api_key
        else:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Could not validate credentials",
            )
    # If API_KEY is not set in environment, security is disabled
    return None


async def validate_url_security(url: str):
    """
    Validates the URL to prevent SSRF attacks (Requirement 9).
    Async-first implementation using non-blocking DNS resolution.
    """
    try:
        parsed_url = urlparse(url)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid URL format.")

    # 1. Check Scheme
    if parsed_url.scheme not in ["http", "https"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid scheme. Only 'http' and 'https' are allowed.",
        )

    hostname = parsed_url.hostname
    if not hostname:
        raise HTTPException(status_code=400, detail="URL must contain a hostname.")

    # 2. Check domain allowlist with subdomain support
    allowed_domains: List[str] = settings.ALLOWED_DOMAINS  # type: ignore[assignment]
    if allowed_domains:
        is_allowed = False
        for allowed_domain in allowed_domains:
            # Allow exact match or subdomain match
            if hostname == allowed_domain or hostname.endswith("." + allowed_domain):
                is_allowed = True
                break

        if not is_allowed:
            raise HTTPException(
                status_code=400, detail=f"Domain '{hostname}' is not in the allowlist."
            )

    # 3. Resolve Hostname (Async DNS resolution for both IPv4 and IPv6)
    try:
        loop = asyncio.get_running_loop()
        # Use AF_UNSPEC to collect both IPv4 and IPv6 addresses
        addr_info = await loop.getaddrinfo(hostname, None, family=socket.AF_UNSPEC)
        if not addr_info:
            raise socket.gaierror("No address info found.")

        # Check all resolved IP addresses
        for addr in addr_info:
            ip_address_str = addr[4][0]

            # 4. Check IP address validity and restrictions
            try:
                ip = ipaddress.ip_address(ip_address_str)
            except ValueError:
                logger.warning(f"Invalid IP address resolved: {ip_address_str}")
                continue

            # Requirement: disallow localhost, link-local, and private IP ranges
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
                logger.warning(
                    f"SSRF attempt detected: URL {url} resolved to forbidden IP {ip_address_str}"
                )
                raise HTTPException(
                    status_code=400,
                    detail="Access to private, loopback, or reserved IP ranges is forbidden.",
                )

    except (socket.gaierror, OSError) as e:
        logger.warning(f"Could not resolve hostname: {hostname}. Error: {e}")
        # Treat DNS resolution failure as a bad request or potential SSRF attempt
        raise HTTPException(status_code=400, detail="Could not resolve hostname.")

    return True


async def validate_redirect_target_security(url: str):
    """
    Revalidate the final redirect target.
    Performs a lightweight async HEAD request to obtain the final URL,
    then reruns the same security checks on the final hostname.
    """
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(5.0), follow_redirects=True
        ) as client:
            response = await client.head(url)
            final_url = str(response.url)

            # If the final URL is different, revalidate it
            if final_url != url:
                logger.info(f"URL {url} redirected to {final_url}, revalidating...")
                await validate_url_security(final_url)

    except (httpx.RequestError, httpx.TimeoutException) as e:
        logger.warning(f"Could not check redirect target for {url}: {e}")
        # If HEAD request fails, proceed with original validated URL
        pass
    except HTTPException:
        # Re-raise security validation failures
        raise

    return True
