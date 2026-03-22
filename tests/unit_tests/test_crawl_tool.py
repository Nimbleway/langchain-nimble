"""Unit tests for NimbleCrawlTool."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from langchain_nimble import NimbleCrawlTool


def _mock_crawl_run_response(crawl_id: str = "crawl-123") -> MagicMock:
    """Create a mock CrawlRunResponse."""
    mock = MagicMock()
    mock.crawl_id = crawl_id
    mock.status = "queued"
    return mock


def _mock_crawl_status_response(
    status: str = "running",
    crawl_id: str = "crawl-123",
    tasks: list[dict[str, object]] | None = None,
) -> MagicMock:
    """Create a mock CrawlStatusResponse."""
    mock = MagicMock()
    mock.crawl_id = crawl_id
    mock.status = status

    if tasks is not None:
        mock_tasks = []
        for t in tasks:
            task_mock = MagicMock()
            task_mock.model_dump.return_value = t
            mock_tasks.append(task_mock)
        mock.tasks = mock_tasks
    else:
        mock.tasks = []

    return mock


def test_nimble_crawl_tool_init() -> None:
    """Test NimbleCrawlTool initialization."""
    tool = NimbleCrawlTool(api_key="test_key")
    assert tool.name == "nimble_crawl"
    assert tool.polling_interval == 5.0
    assert tool.timeout == 300.0
    assert tool._sync_client is not None
    assert tool._async_client is not None


def test_nimble_crawl_tool_custom_polling() -> None:
    """Test NimbleCrawlTool with custom polling settings."""
    tool = NimbleCrawlTool(api_key="test_key", polling_interval=2.0, timeout=60.0)
    assert tool.polling_interval == 2.0
    assert tool.timeout == 60.0


def test_nimble_crawl_tool_missing_api_key() -> None:
    """Test NimbleCrawlTool raises error without API key."""
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(ValueError, match="API key required"),
    ):
        NimbleCrawlTool()


@patch("langchain_nimble.tools.crawl_tool.time.sleep")
def test_nimble_crawl_tool_run_immediate_success(mock_sleep: MagicMock) -> None:
    """Test crawl that succeeds on first poll."""
    tool = NimbleCrawlTool(api_key="test_key", polling_interval=1.0)
    mock_run_resp = _mock_crawl_run_response()
    mock_status_resp = _mock_crawl_status_response(
        status="succeeded",
        tasks=[
            {"url": "https://example.com", "status": "completed", "content": "page1"},
        ],
    )

    with (
        patch.object(tool._sync_client.crawl, "run", return_value=mock_run_resp),
        patch.object(tool._sync_client.crawl, "status", return_value=mock_status_resp),
    ):
        result = tool._run(url="https://example.com")

    assert len(result) == 1
    assert result[0]["url"] == "https://example.com"
    mock_sleep.assert_called_once_with(1.0)


@patch("langchain_nimble.tools.crawl_tool.time.sleep")
def test_nimble_crawl_tool_run_polls_until_success(mock_sleep: MagicMock) -> None:
    """Test crawl that requires multiple polls before succeeding."""
    tool = NimbleCrawlTool(api_key="test_key", polling_interval=1.0)
    mock_run_resp = _mock_crawl_run_response()

    running_resp = _mock_crawl_status_response(status="running")
    success_resp = _mock_crawl_status_response(
        status="succeeded",
        tasks=[{"url": "https://example.com/page1", "content": "data"}],
    )

    with (
        patch.object(tool._sync_client.crawl, "run", return_value=mock_run_resp),
        patch.object(
            tool._sync_client.crawl,
            "status",
            side_effect=[running_resp, running_resp, success_resp],
        ),
    ):
        result = tool._run(url="https://example.com")

    assert len(result) == 1
    assert mock_sleep.call_count == 3


@patch("langchain_nimble.tools.crawl_tool.time.sleep")
def test_nimble_crawl_tool_run_failed(mock_sleep: MagicMock) -> None:
    """Test crawl raises ToolException when job fails."""
    tool = NimbleCrawlTool(api_key="test_key", polling_interval=1.0)
    mock_run_resp = _mock_crawl_run_response()
    failed_resp = _mock_crawl_status_response(status="failed")

    with (
        patch.object(tool._sync_client.crawl, "run", return_value=mock_run_resp),
        patch.object(tool._sync_client.crawl, "status", return_value=failed_resp),
        pytest.raises(ToolException, match="ended with status 'failed'"),
    ):
        tool._run(url="https://example.com")


@patch("langchain_nimble.tools.crawl_tool.time.sleep")
def test_nimble_crawl_tool_run_canceled(mock_sleep: MagicMock) -> None:
    """Test crawl raises ToolException when job is canceled."""
    tool = NimbleCrawlTool(api_key="test_key", polling_interval=1.0)
    mock_run_resp = _mock_crawl_run_response()
    canceled_resp = _mock_crawl_status_response(status="canceled")

    with (
        patch.object(tool._sync_client.crawl, "run", return_value=mock_run_resp),
        patch.object(tool._sync_client.crawl, "status", return_value=canceled_resp),
        pytest.raises(ToolException, match="ended with status 'canceled'"),
    ):
        tool._run(url="https://example.com")


@patch("langchain_nimble.tools.crawl_tool.time.sleep")
def test_nimble_crawl_tool_run_timeout(mock_sleep: MagicMock) -> None:
    """Test crawl raises ToolException on timeout."""
    tool = NimbleCrawlTool(api_key="test_key", polling_interval=1.0, timeout=3.0)
    mock_run_resp = _mock_crawl_run_response()
    running_resp = _mock_crawl_status_response(status="running")

    with (
        patch.object(tool._sync_client.crawl, "run", return_value=mock_run_resp),
        patch.object(tool._sync_client.crawl, "status", return_value=running_resp),
        pytest.raises(ToolException, match="timed out"),
    ):
        tool._run(url="https://example.com")

    # Should have polled 3 times (at 1s, 2s, 3s) before timeout
    assert mock_sleep.call_count == 3


@patch("langchain_nimble.tools.crawl_tool.asyncio.sleep")
async def test_nimble_crawl_tool_arun_basic(mock_sleep: MagicMock) -> None:
    """Test basic asynchronous crawl."""
    tool = NimbleCrawlTool(api_key="test_key", polling_interval=1.0)
    mock_run_resp = _mock_crawl_run_response()
    success_resp = _mock_crawl_status_response(
        status="succeeded",
        tasks=[{"url": "https://example.com", "content": "async data"}],
    )

    with (
        patch.object(tool._async_client.crawl, "run", return_value=mock_run_resp),
        patch.object(tool._async_client.crawl, "status", return_value=success_resp),
    ):
        result = await tool._arun(url="https://example.com")

    assert len(result) == 1
    assert result[0]["content"] == "async data"
    mock_sleep.assert_awaited_once()


def test_nimble_crawl_tool_run_with_options() -> None:
    """Test crawl passes all options to SDK."""
    tool = NimbleCrawlTool(api_key="test_key", polling_interval=1.0)
    mock_run_resp = _mock_crawl_run_response()
    success_resp = _mock_crawl_status_response(status="succeeded", tasks=[])

    with (
        patch("langchain_nimble.tools.crawl_tool.time.sleep"),
        patch.object(
            tool._sync_client.crawl, "run", return_value=mock_run_resp
        ) as mock_run,
        patch.object(tool._sync_client.crawl, "status", return_value=success_resp),
    ):
        tool._run(
            url="https://example.com",
            limit=10,
            max_discovery_depth=2,
            allow_external_links=False,
            include_paths=["/blog/*"],
            exclude_paths=["/admin/*"],
            sitemap="include",
            name="test-crawl",
        )

    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["url"] == "https://example.com"
    assert call_kwargs["limit"] == 10
    assert call_kwargs["max_discovery_depth"] == 2
    assert call_kwargs["allow_external_links"] is False
    assert call_kwargs["include_paths"] == ["/blog/*"]
    assert call_kwargs["exclude_paths"] == ["/admin/*"]
    assert call_kwargs["sitemap"] == "include"
    assert call_kwargs["name"] == "test-crawl"


def test_nimble_crawl_tool_input_validation() -> None:
    """Test NimbleCrawlToolInput validation."""
    from langchain_nimble.tools.crawl_tool import NimbleCrawlToolInput

    valid_input = NimbleCrawlToolInput(url="https://example.com")
    assert valid_input.url == "https://example.com"
    assert valid_input.limit is None

    valid_with_options = NimbleCrawlToolInput(
        url="https://example.com",
        limit=50,
        max_discovery_depth=3,
        include_paths=["/docs/*"],
    )
    assert valid_with_options.limit == 50
    assert valid_with_options.max_discovery_depth == 3
