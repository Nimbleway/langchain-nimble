"""Unit tests for Nimble tools (search and extract)."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from langchain_nimble import NimbleExtractTool, NimbleSearchTool


def test_nimble_search_tool_init() -> None:
    """Test NimbleSearchTool initialization."""
    tool = NimbleSearchTool(api_key="test_key")
    assert tool.name == "nimble_web_search"
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


@patch("langchain_nimble._utilities.create_sync_client")
def test_nimble_search_tool_run_basic(mock_create_client: MagicMock) -> None:
    """Test basic synchronous search."""
    # Mock the client and response
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "body": [
            {
                "page_content": "Test content",
                "metadata": {
                    "title": "Test Title",
                    "url": "https://example.com",
                    "snippet": "Test snippet",
                },
            }
        ]
    }
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response
    mock_create_client.return_value = mock_client

    # Create tool and run search
    tool = NimbleSearchTool(api_key="test_key")
    result = tool._run(query="test query", num_results=3)

    # Verify
    assert result is not None
    assert "body" in result
    mock_client.post.assert_called_once()


@patch("langchain_nimble._utilities.create_async_client")
async def test_nimble_search_tool_arun_basic(mock_create_client: MagicMock) -> None:
    """Test basic asynchronous search."""
    # Mock the async client and response
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "body": [
            {
                "page_content": "Test content",
                "metadata": {
                    "title": "Test Title",
                    "url": "https://example.com",
                    "snippet": "Test snippet",
                },
            }
        ]
    }
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    # Make post method return an awaitable
    async def mock_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        return mock_response

    mock_client.post = mock_post
    mock_create_client.return_value = mock_client

    # Create tool and run search
    tool = NimbleSearchTool(api_key="test_key")
    result = await tool._arun(query="test query", num_results=3)

    # Verify
    assert result is not None
    assert "body" in result


@patch("langchain_nimble._utilities.create_sync_client")
def test_nimble_search_tool_run_with_options(mock_create_client: MagicMock) -> None:
    """Test synchronous search with all options."""
    # Mock the client and response
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json.return_value = {"body": []}
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response
    mock_create_client.return_value = mock_client

    # Create tool and run search with options
    tool = NimbleSearchTool(api_key="test_key")
    result = tool._run(
        query="test query",
        num_results=5,
        deep_search=True,
        topic="news",
        include_domains=["example.com"],
        exclude_domains=["spam.com"],
        start_date="2024-01-01",
        end_date="2024-12-31",
    )

    # Verify
    assert result is not None
    mock_client.post.assert_called_once()

    # Check the request body includes the options
    call_args = mock_client.post.call_args
    request_body = call_args.kwargs["json"]
    assert request_body["query"] == "test query"
    assert request_body["num_results"] == 5
    assert request_body["deep_search"] is True
    assert request_body["topic"] == "news"
    assert request_body["include_domains"] == ["example.com"]
    assert request_body["exclude_domains"] == ["spam.com"]
    assert request_body["start_date"] == "2024-01-01"
    assert request_body["end_date"] == "2024-12-31"


@patch("langchain_nimble._utilities.create_sync_client")
def test_nimble_search_tool_invoke(mock_create_client: MagicMock) -> None:
    """Test tool invoke method."""
    # Mock the client and response
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json.return_value = {"body": []}
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response
    mock_create_client.return_value = mock_client

    # Create tool and invoke
    tool = NimbleSearchTool(api_key="test_key")
    result = tool.invoke({"query": "test query", "num_results": 3})

    # Verify
    assert result is not None
    mock_client.post.assert_called_once()


@patch("langchain_nimble._utilities.create_async_client")
async def test_nimble_search_tool_ainvoke(mock_create_client: MagicMock) -> None:
    """Test tool async invoke method."""
    # Mock the async client and response
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.json.return_value = {"body": []}
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    # Make post method return an awaitable
    async def mock_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        return mock_response

    mock_client.post = mock_post
    mock_create_client.return_value = mock_client

    # Create tool and invoke
    tool = NimbleSearchTool(api_key="test_key")
    result = await tool.ainvoke({"query": "test query", "num_results": 3})

    # Verify
    assert result is not None


def test_nimble_search_tool_input_validation() -> None:
    """Test NimbleSearchToolInput validation."""
    from langchain_nimble.search_tool import NimbleSearchToolInput

    # Valid input
    valid_input = NimbleSearchToolInput(query="test", num_results=10)
    assert valid_input.query == "test"
    assert valid_input.num_results == 10

    # Test bounds on num_results
    with pytest.raises(Exception):  # Pydantic validation error
        NimbleSearchToolInput(query="test", num_results=0)

    with pytest.raises(Exception):  # Pydantic validation error
        NimbleSearchToolInput(query="test", num_results=101)

    # Test mutually exclusive deep_search and include_answer
    with pytest.raises(ValueError, match="deep_search and include_answer cannot both"):
        NimbleSearchToolInput(query="test", deep_search=True, include_answer=True)


@pytest.mark.benchmark
def test_nimble_search_tool_init_time(benchmark):  # type: ignore[no-untyped-def]
    """Benchmark NimbleSearchTool initialization time."""

    def _init_tool() -> None:
        for _ in range(10):
            NimbleSearchTool(api_key="test_key")

    benchmark(_init_tool)


# ============================================================================
# NimbleExtractTool Tests
# ============================================================================


def test_nimble_extract_tool_init() -> None:
    """Test NimbleExtractTool initialization."""
    tool = NimbleExtractTool(api_key="test_key")
    assert tool.name == "nimble_extract_content"
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


@patch("langchain_nimble._utilities.create_sync_client")
def test_nimble_extract_tool_run_basic(mock_create_client: MagicMock) -> None:
    """Test basic synchronous content extraction."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "body": [
            {
                "page_content": "Extracted content from the page",
                "metadata": {
                    "title": "Page Title",
                    "url": "https://example.com",
                },
            }
        ]
    }
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response
    mock_create_client.return_value = mock_client

    tool = NimbleExtractTool(api_key="test_key")
    result = tool._run(urls=["https://example.com"])

    assert result is not None
    assert "body" in result
    mock_client.post.assert_called_once()


@patch("langchain_nimble._utilities.create_async_client")
async def test_nimble_extract_tool_arun_basic(mock_create_client: MagicMock) -> None:
    """Test basic asynchronous content extraction."""
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "body": [
            {
                "page_content": "Extracted content",
                "metadata": {
                    "title": "Test Title",
                    "url": "https://example.com",
                },
            }
        ]
    }
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    async def mock_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        return mock_response

    mock_client.post = mock_post
    mock_create_client.return_value = mock_client

    tool = NimbleExtractTool(api_key="test_key")
    result = await tool._arun(urls=["https://example.com"])

    assert result is not None
    assert "body" in result


@patch("langchain_nimble._utilities.create_sync_client")
def test_nimble_extract_tool_run_multiple_urls(mock_create_client: MagicMock) -> None:
    """Test synchronous extraction with multiple URLs."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json.return_value = {
        "body": [
            {"page_content": "Content 1", "metadata": {"url": "https://site1.com"}},
            {"page_content": "Content 2", "metadata": {"url": "https://site2.com"}},
            {"page_content": "Content 3", "metadata": {"url": "https://site3.com"}},
        ]
    }
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response
    mock_create_client.return_value = mock_client

    tool = NimbleExtractTool(api_key="test_key")
    result = tool._run(
        urls=["https://site1.com", "https://site2.com", "https://site3.com"]
    )

    assert result is not None
    assert "body" in result
    assert len(result["body"]) == 3
    mock_client.post.assert_called_once()

    call_args = mock_client.post.call_args
    request_body = call_args.kwargs["json"]
    assert "links" in request_body
    assert len(request_body["links"]) == 3


@patch("langchain_nimble._utilities.create_sync_client")
def test_nimble_extract_tool_run_with_options(mock_create_client: MagicMock) -> None:
    """Test synchronous extraction with all options."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json.return_value = {"body": []}
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response
    mock_create_client.return_value = mock_client

    tool = NimbleExtractTool(api_key="test_key")
    result = tool._run(
        urls=["https://example.com"],
        driver="vx6",
        wait=3000,
        locale="fr",
        country="FR",
        parsing_type="markdown",
    )

    assert result is not None
    mock_client.post.assert_called_once()

    call_args = mock_client.post.call_args
    request_body = call_args.kwargs["json"]
    assert request_body["links"] == ["https://example.com"]
    assert request_body["driver"] == "vx6"
    assert request_body["wait"] == 3000
    assert request_body["locale"] == "fr"
    assert request_body["country"] == "FR"
    assert request_body["parsing_type"] == "markdown"


@patch("langchain_nimble._utilities.create_sync_client")
def test_nimble_extract_tool_invoke(mock_create_client: MagicMock) -> None:
    """Test tool invoke method."""
    mock_client = MagicMock(spec=httpx.Client)
    mock_response = MagicMock()
    mock_response.json.return_value = {"body": []}
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()
    mock_client.post.return_value = mock_response
    mock_create_client.return_value = mock_client

    tool = NimbleExtractTool(api_key="test_key")
    result = tool.invoke({"urls": ["https://example.com"]})

    assert result is not None
    mock_client.post.assert_called_once()


@patch("langchain_nimble._utilities.create_async_client")
async def test_nimble_extract_tool_ainvoke(mock_create_client: MagicMock) -> None:
    """Test tool async invoke method."""
    mock_client = MagicMock(spec=httpx.AsyncClient)
    mock_response = MagicMock()
    mock_response.json.return_value = {"body": []}
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    async def mock_post(*args, **kwargs):  # type: ignore[no-untyped-def]
        return mock_response

    mock_client.post = mock_post
    mock_create_client.return_value = mock_client

    tool = NimbleExtractTool(api_key="test_key")
    result = await tool.ainvoke({"urls": ["https://example.com"]})

    assert result is not None


def test_nimble_extract_tool_input_validation() -> None:
    """Test NimbleExtractToolInput validation."""
    from langchain_nimble.extract_tool import NimbleExtractToolInput

    # Valid input with single URL
    valid_input = NimbleExtractToolInput(urls=["https://example.com"])
    assert len(valid_input.urls) == 1

    # Valid input with multiple URLs
    valid_input_multi = NimbleExtractToolInput(
        urls=["https://site1.com", "https://site2.com"]
    )
    assert len(valid_input_multi.urls) == 2

    # Test minimum constraint (must have at least 1 URL)
    with pytest.raises(Exception):  # Pydantic validation error
        NimbleExtractToolInput(urls=[])

    # Test maximum constraint (max 20 URLs)
    with pytest.raises(Exception):  # Pydantic validation error
        NimbleExtractToolInput(urls=[f"https://site{i}.com" for i in range(21)])


@pytest.mark.benchmark
def test_nimble_extract_tool_init_time(benchmark):  # type: ignore[no-untyped-def]
    """Benchmark NimbleExtractTool initialization time."""

    def _init_tool() -> None:
        for _ in range(10):
            NimbleExtractTool(api_key="test_key")

    benchmark(_init_tool)
