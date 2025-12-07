"""HTTP client utilities with retry logic and connection pooling."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

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


class _RetryTransport(httpx.HTTPTransport):
    """HTTP transport with retry logic for 5xx errors."""

    def __init__(self, *args: Any, max_retries: int = 2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        """Retry 5xx errors with exponential backoff (1s, 2s, 4s)."""
        for attempt in range(self.max_retries + 1):
            try:
                response = super().handle_request(request)
                if response.status_code < 500 or attempt == self.max_retries:
                    return response
                time.sleep(2.0**attempt)
            except httpx.RequestError:
                if attempt == self.max_retries:
                    raise
                time.sleep(2.0**attempt)
        msg = "Retry loop completed unexpectedly"
        raise RuntimeError(msg)


class _AsyncRetryTransport(httpx.AsyncHTTPTransport):
    """Async HTTP transport with retry logic for 5xx errors."""

    def __init__(self, *args: Any, max_retries: int = 2, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Retry 5xx errors with exponential backoff (1s, 2s, 4s)."""
        for attempt in range(self.max_retries + 1):
            try:
                response = await super().handle_async_request(request)
                if response.status_code < 500 or attempt == self.max_retries:
                    return response
                await asyncio.sleep(2.0**attempt)
            except httpx.RequestError:
                if attempt == self.max_retries:
                    raise
                await asyncio.sleep(2.0**attempt)
        msg = "Retry loop completed unexpectedly"
        raise RuntimeError(msg)


@contextmanager
def handle_api_errors(operation: str = "API request") -> Iterator[None]:
    """Convert httpx exceptions to user-friendly error messages."""
    try:
        yield
    except httpx.HTTPStatusError as e:
        status = e.response.status_code
        if 400 <= status < 500:
            msg = (
                f"Nimble API {operation} failed with client error ({status}): "
                f"{e.response.text}"
            )
        else:
            msg = (
                f"Nimble API {operation} failed with server error ({status}): "
                f"{e.response.text}"
            )
        raise ValueError(msg) from e
    except httpx.TimeoutException as e:
        msg = f"Nimble API {operation} timed out: {e!s}"
        raise ValueError(msg) from e
    except httpx.RequestError as e:
        msg = f"Nimble API {operation} failed with network error: {e!s}"
        raise ValueError(msg) from e


@lru_cache
def create_sync_client(
    *,
    api_key: str,
    base_url: str,
    timeout: float | httpx.Timeout = 100.0,
    max_retries: int = 2,
) -> _SyncHttpxClientWrapper:
    """Create cached HTTP client with connection pooling and retry logic."""
    transport = _RetryTransport(max_retries=max_retries) if max_retries > 0 else None
    return _SyncHttpxClientWrapper(
        base_url=base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Client-Source": "langchain-nimble",
            "Content-Type": "application/json",
        },
        timeout=timeout,
        transport=transport,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    )


@lru_cache
def create_async_client(
    *,
    api_key: str,
    base_url: str,
    timeout: float | httpx.Timeout = 100.0,
    max_retries: int = 2,
) -> _AsyncHttpxClientWrapper:
    """Create cached async HTTP client with connection pooling and retry logic."""
    transport = (
        _AsyncRetryTransport(max_retries=max_retries) if max_retries > 0 else None
    )
    return _AsyncHttpxClientWrapper(
        base_url=base_url,
        headers={
            "Authorization": f"Bearer {api_key}",
            "X-Client-Source": "langchain-nimble",
            "Content-Type": "application/json",
        },
        timeout=timeout,
        transport=transport,
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
    )
