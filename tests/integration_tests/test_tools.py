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
        {"query": "LangChain framework", "max_results": 3, "deep_search": False}
    )

    assert result is not None
    assert "results" in result
    assert len(result["results"]) > 0
    assert len(result["results"]) <= 3

    # Check first result structure (v1.6.0: flattened structure)
    first_result = result["results"][0]
    assert "url" in first_result
    assert first_result["url"].startswith("http")


async def test_nimble_search_async_deep_mode(api_key: str) -> None:
    """Test async search in deep mode with full content extraction."""
    tool = NimbleSearchTool(api_key=api_key)

    result = await tool.ainvoke(
        {"query": "Python programming", "max_results": 2, "deep_search": True}
    )

    assert result is not None
    assert "results" in result
    assert len(result["results"]) > 0
    assert len(result["results"]) <= 2

    # Check that we got full content (v1.6.0: content field at top-level)
    first_result = result["results"][0]
    assert "content" in first_result
    assert len(first_result["content"]) > 0


async def test_nimble_search_async_with_filters(api_key: str) -> None:
    """Test async search with domain filtering in fast mode."""
    tool = NimbleSearchTool(api_key=api_key)

    result = await tool.ainvoke(
        {
            "query": "Python documentation",
            "max_results": 5,
            "deep_search": False,
            "include_domains": ["python.org", "docs.python.org"],
        }
    )

    assert result is not None
    assert "results" in result
    assert len(result["results"]) > 0

    # Verify domain filtering worked (if results returned)
    # v1.6.0: url is at top-level, not nested in metadata
    for item in result["results"]:
        url = item.get("url", "")
        # Some results might match, but we just check structure is correct
        assert url.startswith("http")


async def test_nimble_search_async_news_focus(api_key: str) -> None:
    """Test async search with news focus mode (v1.5.0: renamed from topic)."""
    tool = NimbleSearchTool(api_key=api_key)

    result = await tool.ainvoke(
        {
            "query": "latest technology news",
            "max_results": 3,
            "deep_search": False,
            "focus": "news",
        }
    )

    assert result is not None
    assert "results" in result
    assert len(result["results"]) > 0


async def test_nimble_search_async_invalid_api_key() -> None:
    """Test async error handling for invalid API key in fast mode."""
    tool = NimbleSearchTool(api_key="invalid_key")

    with pytest.raises(Exception):
        await tool.ainvoke(
            {"query": "test query", "max_results": 1, "deep_search": False}
        )


async def test_nimble_search_sync_invoke(api_key: str) -> None:
    """Test synchronous invoke method in fast mode."""
    tool = NimbleSearchTool(api_key=api_key)

    result = tool.invoke({"query": "LangChain", "max_results": 2, "deep_search": False})

    assert result is not None
    assert "results" in result
    assert len(result["results"]) > 0
    assert len(result["results"]) <= 2


# ============================================================================
# NimbleExtractTool Integration Tests
# ============================================================================


async def test_nimble_extract_async_single_url(api_key: str) -> None:
    """Test async content extraction from a single URL."""
    tool = NimbleExtractTool(api_key=api_key)

    result = await tool.ainvoke({"urls": ["https://example.com"]})

    assert result is not None
    assert "results" in result
    assert len(result["results"]) > 0

    # v1.6.0: content field at top-level, not nested
    first_result = result["results"][0]
    assert "content" in first_result
    assert len(first_result["content"]) > 0
