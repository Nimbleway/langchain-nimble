"""HTTP client utilities with retry logic and connection pooling."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager
from functools import lru_cache
from typing import Any

import httpx
from langchain_core.utils import from_env, secret_from_env
from pydantic import BaseModel, Field, SecretStr, model_validator


class _NimbleClientMixin(BaseModel):
    """Mixin providing Nimble API client configuration and initialization.

    This mixin is shared by both retrievers and tools to avoid code duplication
    for client configuration and initialization logic.
    """

    nimble_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env("NIMBLE_API_KEY", default=""),
    )
    nimble_api_url: str = Field(
        alias="base_url",
        default_factory=from_env(
            "NIMBLE_API_URL",
            default="https://nimble-retriever.webit.live",
        ),
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retry attempts for 5xx errors (0 disables retries)",
    )

    locale: str = "en"
    country: str = "US"
    parsing_type: str = "markdown"

    _sync_client: httpx.Client | None = None
    _async_client: httpx.AsyncClient | None = None

    @model_validator(mode="after")
    def initialize_clients(self) -> _NimbleClientMixin:
        """Initialize HTTP clients."""
        api_key = self.nimble_api_key.get_secret_value()
        if not api_key:
            msg = "API key required. Set NIMBLE_API_KEY or pass api_key parameter."
            raise ValueError(msg)

        self._sync_client = create_sync_client(
            api_key=api_key, base_url=self.nimble_api_url, max_retries=self.max_retries
        )
        self._async_client = create_async_client(
            api_key=api_key, base_url=self.nimble_api_url, max_retries=self.max_retries
        )
        return self


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


def create_async_client(
    *,
    api_key: str,
    base_url: str,
    timeout: float | httpx.Timeout = 100.0,
    max_retries: int = 2,
) -> _AsyncHttpxClientWrapper:
    """Create async HTTP client with connection pooling and retry logic.

    Not cached because AsyncClient's connection pool is bound to a specific
    event loop. With pytest-asyncio creating new loops per test, caching
    causes "Event loop is closed" errors when pooled connections from old
    loops are reused.

    Each instance still benefits from connection pooling within its own client.
    See: https://github.com/encode/httpx/discussions/2959
    """
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
