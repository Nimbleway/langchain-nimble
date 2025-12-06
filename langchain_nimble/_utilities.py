"""Helpers for creating Nimble API clients with caching and connection pooling."""

from __future__ import annotations

import asyncio
from functools import lru_cache

import httpx


class _SyncHttpxClientWrapper(httpx.Client):
    """Wrapper around httpx.Client with automatic cleanup."""

    def __del__(self) -> None:
        """Close client on garbage collection."""
        if self.is_closed:
            return

        try:
            self.close()
        except Exception:  # noqa: S110
            pass


class _AsyncHttpxClientWrapper(httpx.AsyncClient):
    """Wrapper around httpx.AsyncClient with automatic cleanup."""

    def __del__(self) -> None:
        """Close client on garbage collection."""
        if self.is_closed:
            return

        try:
            asyncio.get_running_loop().create_task(self.aclose())
        except Exception:  # noqa: S110
            pass


@lru_cache
def create_sync_client(
    *,
    api_key: str,
    base_url: str,
    timeout: float | httpx.Timeout = 100.0,
) -> _SyncHttpxClientWrapper:
    """Create cached sync HTTP client with connection pooling.

    Args:
        api_key: API key for authentication.
        base_url: Base URL for API requests.
        timeout: Request timeout in seconds (default: 100.0).

    Returns:
        Cached httpx.Client with connection pooling and tracking headers.

    Examples:
        >>> client = create_sync_client(
        ...     api_key="my-key", base_url="https://api.example.com"
        ... )
        >>> response = client.post("/search", json={"query": "test"})
    """
    return _SyncHttpxClientWrapper(
        base_url=base_url,
        headers={
            "Authorization": f"Basic {api_key}",
            "X-Client-Source": "langchain-nimble",
            "Content-Type": "application/json",
        },
        timeout=timeout,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    )


@lru_cache
def create_async_client(
    *,
    api_key: str,
    base_url: str,
    timeout: float | httpx.Timeout = 100.0,
) -> _AsyncHttpxClientWrapper:
    """Create cached async HTTP client with connection pooling.

    Args:
        api_key: API key for authentication.
        base_url: Base URL for API requests.
        timeout: Request timeout in seconds (default: 100.0).

    Returns:
        Cached httpx.AsyncClient with connection pooling and tracking headers.

    Examples:
        >>> client = create_async_client(
        ...     api_key="my-key", base_url="https://api.example.com"
        ... )
        >>> response = await client.post("/search", json={"query": "test"})
    """
    return _AsyncHttpxClientWrapper(
        base_url=base_url,
        headers={
            "Authorization": f"Basic {api_key}",
            "X-Client-Source": "langchain-nimble",
            "Content-Type": "application/json",
        },
        timeout=timeout,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    )
