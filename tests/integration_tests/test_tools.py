"""Integration tests for Nimble tools (search and extract).

Requires NIMBLE_API_KEY environment variable.
"""

import os

import pytest

from langchain_nimble import NimbleExtractTool, NimbleSearchTool


@pytest.fixture
def api_key() -> str:
    """Get API key from environment or skip test."""
    key = os.environ.get("NIMBLE_API_KEY")
    if not key:
        pytest.skip("NIMBLE_API_KEY not set")
    return key


async def test_nimble_search_async_fast_mode(api_key: str) -> None:
    """Test async search in fast mode (deep_search=False)."""
    tool = NimbleSearchTool(api_key=api_key)

    result = await tool.ainvoke(
        {"query": "LangChain framework", "num_results": 3, "deep_search": False}
    )

    assert result is not None
    assert "body" in result
    assert len(result["body"]) > 0
    assert len(result["body"]) <= 3

    # Check first result structure (metadata only, no full content)
    first_result = result["body"][0]
    assert "metadata" in first_result
    assert "url" in first_result["metadata"]
    assert first_result["metadata"]["url"].startswith("http")


async def test_nimble_search_async_deep_mode(api_key: str) -> None:
    """Test async search in deep mode with full content extraction."""
    tool = NimbleSearchTool(api_key=api_key)

    result = await tool.ainvoke(
        {"query": "Python programming", "num_results": 2, "deep_search": True}
    )

    assert result is not None
    assert "body" in result
    assert len(result["body"]) > 0
    assert len(result["body"]) <= 2

    # Check that we got full content
    first_result = result["body"][0]
    assert "page_content" in first_result
    assert len(first_result["page_content"]) > 0


async def test_nimble_search_async_with_filters(api_key: str) -> None:
    """Test async search with domain filtering in fast mode."""
    tool = NimbleSearchTool(api_key=api_key)

    result = await tool.ainvoke(
        {
            "query": "Python documentation",
            "num_results": 5,
            "deep_search": False,
            "include_domains": ["python.org", "docs.python.org"],
        }
    )

    assert result is not None
    assert "body" in result
    assert len(result["body"]) > 0

    # Verify domain filtering worked (if results returned)
    for item in result["body"]:
        url = item.get("metadata", {}).get("url", "")
        # Some results might match, but we just check structure is correct
        assert url.startswith("http")


async def test_nimble_search_async_news_topic(api_key: str) -> None:
    """Test async search with news topic in fast mode."""
    tool = NimbleSearchTool(api_key=api_key)

    result = await tool.ainvoke(
        {
            "query": "latest technology news",
            "num_results": 3,
            "deep_search": False,
            "topic": "news",
        }
    )

    assert result is not None
    assert "body" in result
    assert len(result["body"]) > 0


async def test_nimble_search_async_invalid_api_key() -> None:
    """Test async error handling for invalid API key in fast mode."""
    tool = NimbleSearchTool(api_key="invalid_key")

    with pytest.raises(Exception):
        await tool.ainvoke(
            {"query": "test query", "num_results": 1, "deep_search": False}
        )


async def test_nimble_search_sync_invoke(api_key: str) -> None:
    """Test synchronous invoke method in fast mode."""
    tool = NimbleSearchTool(api_key=api_key)

    result = tool.invoke({"query": "LangChain", "num_results": 2, "deep_search": False})

    assert result is not None
    assert "body" in result
    assert len(result["body"]) > 0
    assert len(result["body"]) <= 2


# ============================================================================
# NimbleExtractTool Integration Tests
# ============================================================================


async def test_nimble_extract_async_single_url(api_key: str) -> None:
    """Test async content extraction from a single URL."""
    tool = NimbleExtractTool(api_key=api_key)

    result = await tool.ainvoke({"urls": ["https://example.com"]})

    assert result is not None
    assert "body" in result
    assert len(result["body"]) > 0

    first_result = result["body"][0]
    assert "page_content" in first_result
    assert len(first_result["page_content"]) > 0
    assert "metadata" in first_result
