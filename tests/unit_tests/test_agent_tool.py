"""Unit tests for deprecated NimbleAgent* aliases (Extract Templates)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from langchain_nimble import NimbleAgentGetTool, NimbleAgentListTool, NimbleAgentRunTool


def _mock_template_list_response() -> MagicMock:
    """Create a mock TemplateListResponse with .items."""
    item1 = MagicMock()
    item1.model_dump.return_value = {
        "name": "amazon_pdp",
        "display_name": "Amazon Product Page",
    }
    item2 = MagicMock()
    item2.model_dump.return_value = {
        "name": "google_search",
        "display_name": "Google Search",
    }
    response = MagicMock()
    response.items = [item1, item2]
    return response


def test_nimble_agent_list_tool_init() -> None:
    """Test NimbleAgentListTool initialization."""
    tool = NimbleAgentListTool(api_key="test_key")
    assert tool.name == "nimble_agent_list"


def test_nimble_agent_list_tool_run_basic() -> None:
    """Test deprecated list alias calls extract.templates.list."""
    tool = NimbleAgentListTool(api_key="test_key")
    mock_response = _mock_template_list_response()

    with (
        patch.object(
            tool._sync_client.extract.templates,
            "list",
            return_value=mock_response,
        ) as mock_list,
        pytest.warns(DeprecationWarning, match="Extract Templates"),
    ):
        result = tool._run()

    assert len(result) == 2
    assert result[0]["name"] == "amazon_pdp"
    mock_list.assert_called_once()


async def test_nimble_agent_list_tool_arun_basic() -> None:
    """Test deprecated async list alias."""
    tool = NimbleAgentListTool(api_key="test_key")
    mock_response = _mock_template_list_response()

    with (
        patch.object(
            tool._async_client.extract.templates,
            "list",
            return_value=mock_response,
        ) as mock_list,
        pytest.warns(DeprecationWarning, match="Extract Templates"),
    ):
        result = await tool._arun(limit=5)

    assert len(result) == 2
    mock_list.assert_awaited_once_with(limit=5)


def test_nimble_agent_list_tool_with_pagination() -> None:
    """Test deprecated list alias pagination kwargs."""
    tool = NimbleAgentListTool(api_key="test_key")
    mock_response = _mock_template_list_response()

    with (
        patch.object(
            tool._sync_client.extract.templates,
            "list",
            return_value=mock_response,
        ) as mock_list,
        pytest.warns(DeprecationWarning),
    ):
        tool._run(limit=10, offset=2)

    call_kwargs = mock_list.call_args.kwargs
    assert call_kwargs["limit"] == 10
    assert call_kwargs["offset"] == 2


def test_nimble_agent_get_tool_run() -> None:
    """Test deprecated get alias calls extract.templates.get."""
    tool = NimbleAgentGetTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"name": "amazon_pdp"}

    with (
        patch.object(
            tool._sync_client.extract.templates,
            "get",
            return_value=mock_response,
        ) as mock_get,
        pytest.warns(DeprecationWarning),
    ):
        result = tool._run(template_name="amazon_pdp")

    assert result["name"] == "amazon_pdp"
    mock_get.assert_called_once_with("amazon_pdp")


async def test_nimble_agent_get_tool_arun() -> None:
    """Test deprecated async get alias."""
    tool = NimbleAgentGetTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"name": "amazon_pdp"}

    with (
        patch.object(
            tool._async_client.extract.templates,
            "get",
            return_value=mock_response,
        ) as mock_get,
        pytest.warns(DeprecationWarning),
    ):
        result = await tool._arun(template_name="amazon_pdp")

    assert result["name"] == "amazon_pdp"
    mock_get.assert_awaited_once()


def _mock_run_response(*, status: str = "success") -> MagicMock:
    """Create a mock TemplateRunResponse."""
    mock = MagicMock()
    mock.status = status
    mock.task_id = "task_123"
    mock.warnings = None
    mock.model_dump.return_value = {"status": status, "task_id": "task_123"}
    return mock


def test_nimble_agent_run_tool_maps_agent_to_template() -> None:
    """Test deprecated run alias maps agent= to template=."""
    tool = NimbleAgentRunTool(api_key="test_key")
    mock_response = _mock_run_response()

    with (
        patch.object(
            tool._sync_client.extract.templates,
            "run",
            return_value=mock_response,
        ) as mock_run,
        pytest.warns(DeprecationWarning),
    ):
        result = tool._run(agent="google_search", params={"query": "ai"})

    assert result["status"] == "success"
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["template"] == "google_search"
    assert call_kwargs["params"] == {"query": "ai"}
    assert "no_html" not in call_kwargs["params"]


async def test_nimble_agent_run_tool_arun() -> None:
    """Test deprecated async run alias."""
    tool = NimbleAgentRunTool(api_key="test_key")
    mock_response = _mock_run_response()

    with (
        patch.object(
            tool._async_client.extract.templates,
            "run",
            return_value=mock_response,
        ) as mock_run,
        pytest.warns(DeprecationWarning),
    ):
        result = await tool._arun(
            agent="google_search",
            params={"query": "ai"},
            localization=True,
        )

    assert result["status"] == "success"
    assert mock_run.call_args.kwargs["localization"] is True


def test_nimble_agent_run_tool_failure_status() -> None:
    """Test deprecated run alias raises on non-success status."""
    tool = NimbleAgentRunTool(api_key="test_key")
    mock_response = _mock_run_response(status="error")

    with (
        patch.object(
            tool._sync_client.extract.templates,
            "run",
            return_value=mock_response,
        ),
        pytest.warns(DeprecationWarning),
        pytest.raises(ToolException, match="error"),
    ):
        tool._run(agent="google_search", params={"query": "ai"})


def test_nimble_agent_run_tool_input_validation() -> None:
    """Test NimbleAgentRunToolInput still accepts agent + params."""
    from langchain_nimble.tools.agent_tool import NimbleAgentRunToolInput

    valid_input = NimbleAgentRunToolInput(
        agent="google_search",
        params={"query": "test"},
    )
    assert valid_input.agent == "google_search"
