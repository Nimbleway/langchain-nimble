"""Unit tests for NimbleMapTool."""

from unittest.mock import MagicMock, patch

import pytest

from langchain_nimble import NimbleMapTool


def _mock_map_response(**overrides: object) -> MagicMock:
    """Create a mock MapResponse."""
    mock = MagicMock()
    mock.model_dump.return_value = {
        "success": True,
        "task_id": "test-task-id",
        "links": [
            {
                "url": "https://example.com/page1",
                "title": "Page 1",
                "description": "First page",
            },
            {
                "url": "https://example.com/page2",
                "title": "Page 2",
                "description": None,
            },
        ],
        **overrides,
    }
    return mock


def test_nimble_map_tool_init() -> None:
    """Test NimbleMapTool initialization."""
    tool = NimbleMapTool(api_key="test_key")
    assert tool.name == "nimble_map"
    assert tool.nimble_api_key.get_secret_value() == "test_key"
    assert tool._sync_client is not None
    assert tool._async_client is not None


def test_nimble_map_tool_missing_api_key() -> None:
    """Test NimbleMapTool raises error without API key."""
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(ValueError, match="API key required"),
    ):
        NimbleMapTool()


def test_nimble_map_tool_run_basic() -> None:
    """Test basic synchronous map."""
    tool = NimbleMapTool(api_key="test_key")
    mock_response = _mock_map_response()

    with patch.object(tool._sync_client, "map", return_value=mock_response) as mock_map:
        result = tool._run(url="https://example.com")

    assert result["success"] is True
    assert len(result["links"]) == 2
    mock_map.assert_called_once()


async def test_nimble_map_tool_arun_basic() -> None:
    """Test basic asynchronous map."""
    tool = NimbleMapTool(api_key="test_key")
    mock_response = _mock_map_response()

    with patch.object(
        tool._async_client, "map", return_value=mock_response
    ) as mock_map:
        result = await tool._arun(url="https://example.com")

    assert result["success"] is True
    assert len(result["links"]) == 2
    mock_map.assert_awaited_once()


def test_nimble_map_tool_run_with_options() -> None:
    """Test synchronous map with all options."""
    tool = NimbleMapTool(api_key="test_key")
    mock_response = _mock_map_response(links=[])

    with patch.object(tool._sync_client, "map", return_value=mock_response) as mock_map:
        result = tool._run(
            url="https://example.com",
            limit=50,
            domain_filter="domain",
            sitemap="include",
            locale="fr",
            country="FR",
        )

    assert result is not None
    mock_map.assert_called_once()

    call_kwargs = mock_map.call_args.kwargs
    assert call_kwargs["url"] == "https://example.com"
    assert call_kwargs["limit"] == 50
    assert call_kwargs["domain_filter"] == "domain"
    assert call_kwargs["sitemap"] == "include"
    assert call_kwargs["locale"] == "fr"
    assert call_kwargs["country"] == "FR"


def test_nimble_map_tool_invoke() -> None:
    """Test tool invoke method."""
    tool = NimbleMapTool(api_key="test_key")
    mock_response = _mock_map_response()

    with patch.object(tool._sync_client, "map", return_value=mock_response) as mock_map:
        result = tool.invoke({"url": "https://example.com"})

    assert result is not None
    mock_map.assert_called_once()


async def test_nimble_map_tool_ainvoke() -> None:
    """Test tool async invoke method."""
    tool = NimbleMapTool(api_key="test_key")
    mock_response = _mock_map_response()

    with patch.object(
        tool._async_client, "map", return_value=mock_response
    ) as mock_map:
        result = await tool.ainvoke({"url": "https://example.com"})

    assert result is not None
    mock_map.assert_awaited_once()


def test_nimble_map_tool_input_validation() -> None:
    """Test NimbleMapToolInput validation."""
    from langchain_nimble.tools.map_tool import NimbleMapToolInput

    valid_input = NimbleMapToolInput(url="https://example.com")
    assert valid_input.url == "https://example.com"
    assert valid_input.limit is None
    assert valid_input.domain_filter is None

    valid_with_options = NimbleMapToolInput(
        url="https://example.com",
        limit=100,
        domain_filter="domain",
        sitemap="only",
    )
    assert valid_with_options.limit == 100
    assert valid_with_options.domain_filter == "domain"
