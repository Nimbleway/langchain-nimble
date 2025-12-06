"""Unit tests for client utility functions."""

from langchain_nimble._utilities import (
    _AsyncHttpxClientWrapper,
    _SyncHttpxClientWrapper,
    create_async_client,
    create_sync_client,
)


def test_sync_client_creation() -> None:
    """Test sync client creation."""
    client = create_sync_client(
        api_key="test-key",
        base_url="https://api.example.com",
    )

    assert isinstance(client, _SyncHttpxClientWrapper)
    assert str(client.base_url) == "https://api.example.com"
    assert client.headers["Authorization"] == "Bearer test-key"
    assert client.headers["X-Client-Source"] == "langchain-nimble"
    assert client.headers["Content-Type"] == "application/json"
    assert client.timeout.read == 100.0


async def test_async_client_creation() -> None:
    """Test async client creation."""
    client = create_async_client(
        api_key="test-key",
        base_url="https://api.example.com",
    )

    assert isinstance(client, _AsyncHttpxClientWrapper)
    assert str(client.base_url) == "https://api.example.com"
    assert client.headers["Authorization"] == "Bearer test-key"
    assert client.headers["X-Client-Source"] == "langchain-nimble"
    assert client.headers["Content-Type"] == "application/json"
    assert client.timeout.read == 100.0
