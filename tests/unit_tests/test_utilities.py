"""Unit tests for client utility functions."""

from nimble_python import AsyncNimble, Nimble

from langchain_nimble._utilities import _NimbleClientMixin


def test_sync_client_creation() -> None:
    """Test sync client is a Nimble SDK instance."""
    mixin = _NimbleClientMixin(api_key="test-key")

    assert isinstance(mixin._sync_client, Nimble)


async def test_async_client_creation() -> None:
    """Test async client is an AsyncNimble SDK instance."""
    mixin = _NimbleClientMixin(api_key="test-key")

    assert isinstance(mixin._async_client, AsyncNimble)


def test_custom_base_url() -> None:
    """Test base URL override."""
    mixin = _NimbleClientMixin(
        api_key="test-key",
        base_url="https://custom.api.com",
    )

    assert mixin.nimble_api_url == "https://custom.api.com"
    assert isinstance(mixin._sync_client, Nimble)


def test_max_retries_config() -> None:
    """Test max_retries is passed to clients."""
    mixin = _NimbleClientMixin(api_key="test-key", max_retries=3)

    assert mixin.max_retries == 3
    assert isinstance(mixin._sync_client, Nimble)
