"""Unit tests for Nimble tools (search and extract)."""

from unittest.mock import MagicMock, patch

import pytest
from nimble_python.types import SearchResponse
from nimble_python.types.search_response import (
    Result,
    ResultMetadataSerpMetadata,
)

from langchain_nimble import NimbleExtractTool, NimbleSearchTool


def _mock_search_response(**overrides: object) -> SearchResponse:
    """Create a mock SearchResponse."""
    defaults: dict[str, object] = {
        "request_id": "test-request-id",
        "results": [
            Result(
                title="Test Title",
                url="https://example.com",
                description="Test description",
                content="Test content",
                metadata=ResultMetadataSerpMetadata(
                    position=1,
                    entity_type="organic",
                    country="US",
                    locale="en",
                ),
            )
        ],
        "total_results": 1,
    }
    defaults.update(overrides)
    return SearchResponse(**defaults)


def test_nimble_search_tool_init() -> None:
    """Test NimbleSearchTool initialization."""
    tool = NimbleSearchTool(api_key="test_key")
    assert tool.name == "nimble_search"
    assert tool.nimble_api_key.get_secret_value() == "test_key"
    assert tool._sync_client is not None
    assert tool._async_client is not None


def test_nimble_search_tool_missing_api_key() -> None:
    """Test NimbleSearchTool raises error without API key."""
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(ValueError, match="API key required"),
    ):
        NimbleSearchTool()


def test_nimble_search_tool_run_basic() -> None:
    """Test basic synchronous search."""
    tool = NimbleSearchTool(api_key="test_key")
    mock_response = _mock_search_response()

    with patch.object(
        tool._sync_client, "search", return_value=mock_response
    ) as mock_search:
        result = tool._run(query="test query", max_results=3)

    assert result is not None
    assert "results" in result
    mock_search.assert_called_once()


async def test_nimble_search_tool_arun_basic() -> None:
    """Test basic asynchronous search."""
    tool = NimbleSearchTool(api_key="test_key")
    mock_response = _mock_search_response()

    with patch.object(
        tool._async_client, "search", return_value=mock_response
    ) as mock_search:
        result = await tool._arun(query="test query", max_results=3)

    assert result is not None
    assert "results" in result
    mock_search.assert_awaited_once()


def test_nimble_search_tool_run_with_options() -> None:
    """Test synchronous search with all options."""
    tool = NimbleSearchTool(api_key="test_key")
    mock_response = _mock_search_response(results=[])

    with patch.object(
        tool._sync_client, "search", return_value=mock_response
    ) as mock_search:
        result = tool._run(
            query="test query",
            max_results=5,
            search_depth="deep",
            focus="news",
            include_domains=["example.com"],
            exclude_domains=["spam.com"],
            start_date="2024-01-01",
            end_date="2024-12-31",
        )

    assert result is not None
    mock_search.assert_called_once()

    call_kwargs = mock_search.call_args.kwargs
    assert call_kwargs["query"] == "test query"
    assert call_kwargs["max_results"] == 5
    assert call_kwargs["search_depth"] == "deep"
    assert call_kwargs["focus"] == "news"
    assert call_kwargs["include_domains"] == ["example.com"]
    assert call_kwargs["exclude_domains"] == ["spam.com"]
    assert call_kwargs["start_date"] == "2024-01-01"
    assert call_kwargs["end_date"] == "2024-12-31"


def test_nimble_search_tool_time_range() -> None:
    """Test search with time_range parameter."""
    tool = NimbleSearchTool(api_key="test_key")
    mock_response = _mock_search_response(results=[])

    with patch.object(
        tool._sync_client, "search", return_value=mock_response
    ) as mock_search:
        result = tool._run(
            query="latest AI breakthroughs",
            max_results=10,
            time_range="week",
        )

    assert result is not None
    mock_search.assert_called_once()

    call_kwargs = mock_search.call_args.kwargs
    assert call_kwargs["time_range"] == "week"
    assert call_kwargs["query"] == "latest AI breakthroughs"


def test_nimble_search_tool_new_focus_modes() -> None:
    """Test search with WSA focus modes (shopping, geo, social)."""
    tool = NimbleSearchTool(api_key="test_key")
    mock_response = _mock_search_response()

    with patch.object(
        tool._sync_client, "search", return_value=mock_response
    ) as mock_search:
        result = tool._run(
            query="best laptop for developers",
            focus="shopping",
            max_results=10,
        )

    assert result is not None
    assert "results" in result
    mock_search.assert_called_once()

    call_kwargs = mock_search.call_args.kwargs
    assert call_kwargs["focus"] == "shopping"


def test_nimble_search_tool_invoke() -> None:
    """Test tool invoke method."""
    tool = NimbleSearchTool(api_key="test_key")
    mock_response = _mock_search_response(results=[])

    with patch.object(
        tool._sync_client, "search", return_value=mock_response
    ) as mock_search:
        result = tool.invoke({"query": "test query", "max_results": 3})

    assert result is not None
    mock_search.assert_called_once()


async def test_nimble_search_tool_ainvoke() -> None:
    """Test tool async invoke method."""
    tool = NimbleSearchTool(api_key="test_key")
    mock_response = _mock_search_response(results=[])

    with patch.object(
        tool._async_client, "search", return_value=mock_response
    ) as mock_search:
        result = await tool.ainvoke({"query": "test query", "max_results": 3})

    assert result is not None
    mock_search.assert_awaited_once()


def test_nimble_search_tool_input_validation() -> None:
    """Test NimbleSearchToolInput validation."""
    from langchain_nimble.search_tool import NimbleSearchToolInput

    # Valid input with max_results
    valid_input = NimbleSearchToolInput(query="test", max_results=10)
    assert valid_input.query == "test"
    assert valid_input.max_results == 10

    # Test bounds on max_results
    with pytest.raises(Exception):  # Pydantic validation error
        NimbleSearchToolInput(query="test", max_results=0)

    with pytest.raises(Exception):  # Pydantic validation error
        NimbleSearchToolInput(query="test", max_results=101)

    # include_answer works with all search_depth levels
    valid = NimbleSearchToolInput(
        query="test",
        search_depth="deep",
        include_answer=True,
    )
    assert valid.include_answer is True


def test_nimble_search_tool_backward_compatibility() -> None:
    """Test that num_results alias still works."""
    from langchain_nimble.search_tool import NimbleSearchToolInput

    # num_results alias should still work
    input_with_alias = NimbleSearchToolInput(query="test", num_results=5)
    assert input_with_alias.max_results == 5

    # Verify it serializes correctly for API
    dumped = input_with_alias.model_dump(by_alias=True)
    assert dumped["num_results"] == 5
    assert "max_results" not in dumped


@pytest.mark.benchmark
def test_nimble_search_tool_init_time(benchmark):  # type: ignore[no-untyped-def]
    """Benchmark NimbleSearchTool initialization time."""

    def _init_tool() -> None:
        for _ in range(10):
            NimbleSearchTool(api_key="test_key")

    benchmark(_init_tool)


# ===========================================================
# NimbleExtractTool Tests
# ===========================================================


def _mock_extract_sdk_response() -> MagicMock:
    """Create a mock SDK ExtractResponse with .data.markdown."""
    mock = MagicMock()
    mock.data.markdown = "# Page Title\n\nExtracted content"
    return mock


def test_nimble_extract_tool_init() -> None:
    """Test NimbleExtractTool initialization."""
    tool = NimbleExtractTool(api_key="test_key")
    assert tool.name == "nimble_extract"
    assert tool.nimble_api_key.get_secret_value() == "test_key"
    assert tool._sync_client is not None
    assert tool._async_client is not None


def test_nimble_extract_tool_missing_api_key() -> None:
    """Test NimbleExtractTool raises error without API key."""
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(ValueError, match="API key required"),
    ):
        NimbleExtractTool()


def test_nimble_extract_tool_run_basic() -> None:
    """Test basic synchronous content extraction returns markdown."""
    tool = NimbleExtractTool(api_key="test_key")
    mock_response = _mock_extract_sdk_response()

    with patch.object(
        tool._sync_client, "extract", return_value=mock_response
    ) as mock_extract:
        result = tool._run(url="https://example.com")

    assert result == "# Page Title\n\nExtracted content"
    mock_extract.assert_called_once()
    call_kwargs = mock_extract.call_args.kwargs
    assert call_kwargs["url"] == "https://example.com"
    assert call_kwargs["formats"] == ["markdown"]


async def test_nimble_extract_tool_arun_basic() -> None:
    """Test basic asynchronous content extraction returns markdown."""
    tool = NimbleExtractTool(api_key="test_key")
    mock_response = _mock_extract_sdk_response()

    with patch.object(
        tool._async_client,
        "extract",
        return_value=mock_response,
    ) as mock_extract:
        result = await tool._arun(url="https://example.com")

    assert result == "# Page Title\n\nExtracted content"
    mock_extract.assert_awaited_once()


def test_nimble_extract_tool_invoke() -> None:
    """Test tool invoke method."""
    tool = NimbleExtractTool(api_key="test_key")
    mock_response = _mock_extract_sdk_response()

    with patch.object(
        tool._sync_client, "extract", return_value=mock_response
    ) as mock_extract:
        result = tool.invoke({"url": "https://example.com"})

    assert isinstance(result, str)
    assert "Extracted content" in result
    mock_extract.assert_called_once()


async def test_nimble_extract_tool_ainvoke() -> None:
    """Test tool async invoke method."""
    tool = NimbleExtractTool(api_key="test_key")
    mock_response = _mock_extract_sdk_response()

    with patch.object(
        tool._async_client,
        "extract",
        return_value=mock_response,
    ) as mock_extract:
        result = await tool.ainvoke({"url": "https://example.com"})

    assert isinstance(result, str)
    assert "Extracted content" in result
    mock_extract.assert_awaited_once()


def test_nimble_extract_tool_empty_response() -> None:
    """Test extraction returns empty string when no markdown."""
    tool = NimbleExtractTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.data.markdown = None

    with patch.object(tool._sync_client, "extract", return_value=mock_response):
        result = tool._run(url="https://example.com")

    assert result == ""


def test_nimble_extract_tool_input_validation() -> None:
    """Test NimbleExtractToolInput validation."""
    from langchain_nimble.extract_tool import NimbleExtractToolInput

    valid_input = NimbleExtractToolInput(url="https://example.com")
    assert valid_input.url == "https://example.com"


@pytest.mark.benchmark
def test_nimble_extract_tool_init_time(benchmark):  # type: ignore[no-untyped-def]
    """Benchmark NimbleExtractTool initialization time."""

    def _init_tool() -> None:
        for _ in range(10):
            NimbleExtractTool(api_key="test_key")

    benchmark(_init_tool)
