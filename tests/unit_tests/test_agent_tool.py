"""Unit tests for Nimble Agent tools (list, get, run)."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from langchain_nimble import NimbleAgentGetTool, NimbleAgentListTool, NimbleAgentRunTool

# ───────────────────────────────────────────────────────────────
# NimbleAgentListTool Tests
# ───────────────────────────────────────────────────────────────


def _mock_agent_list_response() -> list[MagicMock]:
    """Create a mock agent list response."""
    agent1 = MagicMock()
    agent1.model_dump.return_value = {
        "name": "amazon_pdp",
        "display_name": "Amazon Product Page",
        "is_public": True,
        "managed_by": "nimble",
        "description": "Extract Amazon product data",
    }
    agent2 = MagicMock()
    agent2.model_dump.return_value = {
        "name": "google_search",
        "display_name": "Google Search",
        "is_public": True,
        "managed_by": "nimble",
        "description": "Extract Google search results",
    }
    return [agent1, agent2]


def test_nimble_agent_list_tool_init() -> None:
    """Test NimbleAgentListTool initialization."""
    tool = NimbleAgentListTool(api_key="test_key")
    assert tool.name == "nimble_agent_list"


def test_nimble_agent_list_tool_run_basic() -> None:
    """Test basic synchronous agent list."""
    tool = NimbleAgentListTool(api_key="test_key")
    mock_response = _mock_agent_list_response()

    with patch.object(
        tool._sync_client.agent, "list", return_value=mock_response
    ) as mock_list:
        result = tool._run()

    assert len(result) == 2
    assert result[0]["name"] == "amazon_pdp"
    assert result[1]["name"] == "google_search"
    mock_list.assert_called_once()


async def test_nimble_agent_list_tool_arun_basic() -> None:
    """Test basic asynchronous agent list."""
    tool = NimbleAgentListTool(api_key="test_key")
    mock_response = _mock_agent_list_response()

    with patch.object(
        tool._async_client.agent, "list", return_value=mock_response
    ) as mock_list:
        result = await tool._arun()

    assert len(result) == 2
    mock_list.assert_awaited_once()


def test_nimble_agent_list_tool_with_filters() -> None:
    """Test agent list with filter parameters."""
    tool = NimbleAgentListTool(api_key="test_key")
    mock_response = _mock_agent_list_response()

    with patch.object(
        tool._sync_client.agent, "list", return_value=mock_response
    ) as mock_list:
        tool._run(
            search="amazon",
            managed_by="nimble",
            privacy="public",
            limit=10,
        )

    call_kwargs = mock_list.call_args.kwargs
    assert call_kwargs["search"] == "amazon"
    assert call_kwargs["managed_by"] == "nimble"
    assert call_kwargs["privacy"] == "public"
    assert call_kwargs["limit"] == 10


# ───────────────────────────────────────────────────────────────
# NimbleAgentGetTool Tests
# ───────────────────────────────────────────────────────────────


def _mock_agent_get_response() -> MagicMock:
    """Create a mock AgentGetResponse."""
    mock = MagicMock()
    mock.model_dump.return_value = {
        "name": "amazon_pdp",
        "display_name": "Amazon Product Page",
        "is_public": True,
        "description": "Extract Amazon product data",
        "input_properties": [
            {
                "name": "asin",
                "type": "string",
                "required": True,
                "description": "Amazon product ASIN",
                "examples": ["B0D1234567"],
            },
            {
                "name": "zip_code",
                "type": "string",
                "required": False,
                "description": "ZIP code for localization",
            },
        ],
        "output_schema": {"type": "object"},
    }
    return mock


def test_nimble_agent_get_tool_init() -> None:
    """Test NimbleAgentGetTool initialization."""
    tool = NimbleAgentGetTool(api_key="test_key")
    assert tool.name == "nimble_agent_get"


def test_nimble_agent_get_tool_run_basic() -> None:
    """Test basic synchronous agent get."""
    tool = NimbleAgentGetTool(api_key="test_key")
    mock_response = _mock_agent_get_response()

    with patch.object(
        tool._sync_client.agent, "get", return_value=mock_response
    ) as mock_get:
        result = tool._run(template_name="amazon_pdp")

    assert result["name"] == "amazon_pdp"
    assert len(result["input_properties"]) == 2
    assert result["input_properties"][0]["name"] == "asin"
    assert result["input_properties"][0]["required"] is True
    mock_get.assert_called_once_with("amazon_pdp")


async def test_nimble_agent_get_tool_arun_basic() -> None:
    """Test basic asynchronous agent get."""
    tool = NimbleAgentGetTool(api_key="test_key")
    mock_response = _mock_agent_get_response()

    with patch.object(
        tool._async_client.agent, "get", return_value=mock_response
    ) as mock_get:
        result = await tool._arun(template_name="amazon_pdp")

    assert result["name"] == "amazon_pdp"
    mock_get.assert_awaited_once_with("amazon_pdp")


def test_nimble_agent_get_tool_invoke() -> None:
    """Test tool invoke method."""
    tool = NimbleAgentGetTool(api_key="test_key")
    mock_response = _mock_agent_get_response()

    with patch.object(
        tool._sync_client.agent, "get", return_value=mock_response
    ) as mock_get:
        result = tool.invoke({"template_name": "amazon_pdp"})

    assert result is not None
    mock_get.assert_called_once()


# ───────────────────────────────────────────────────────────────
# NimbleAgentRunTool Tests
# ───────────────────────────────────────────────────────────────


def _mock_agent_run_response(status: str = "success", **overrides: object) -> MagicMock:
    """Create a mock AgentRunResponse."""
    mock = MagicMock()
    mock.status = status
    mock.task_id = "test-task-id"
    mock.url = "https://example.com"
    mock.warnings = None
    mock.model_dump.return_value = {
        "status": status,
        "task_id": "test-task-id",
        "url": "https://example.com",
        "data": {"markdown": "# Test Content", "parsing": {"key": "value"}},
        "metadata": {},
        **overrides,
    }
    for key, value in overrides.items():
        setattr(mock, key, value)
    return mock


def test_nimble_agent_run_tool_init() -> None:
    """Test NimbleAgentRunTool initialization."""
    tool = NimbleAgentRunTool(api_key="test_key")
    assert tool.name == "nimble_agent_run"
    assert tool.nimble_api_key.get_secret_value() == "test_key"
    assert tool._sync_client is not None
    assert tool._async_client is not None


def test_nimble_agent_run_tool_missing_api_key() -> None:
    """Test NimbleAgentRunTool raises error without API key."""
    with (
        patch.dict("os.environ", {}, clear=True),
        pytest.raises(ValueError, match="API key required"),
    ):
        NimbleAgentRunTool()


def test_nimble_agent_run_tool_run_basic() -> None:
    """Test basic synchronous agent run."""
    tool = NimbleAgentRunTool(api_key="test_key")
    mock_response = _mock_agent_run_response()

    with patch.object(
        tool._sync_client.agent, "run", return_value=mock_response
    ) as mock_run:
        result = tool._run(agent="google_search", params={"query": "test"})

    assert result["status"] == "success"
    assert result["data"]["markdown"] == "# Test Content"
    mock_run.assert_called_once()

    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["agent"] == "google_search"
    assert call_kwargs["params"] == {"no_html": True, "query": "test"}


async def test_nimble_agent_run_tool_arun_basic() -> None:
    """Test basic asynchronous agent run."""
    tool = NimbleAgentRunTool(api_key="test_key")
    mock_response = _mock_agent_run_response()

    with patch.object(
        tool._async_client.agent, "run", return_value=mock_response
    ) as mock_run:
        result = await tool._arun(agent="google_search", params={"query": "test"})

    assert result["status"] == "success"
    mock_run.assert_awaited_once()


def test_nimble_agent_run_tool_error_status() -> None:
    """Test agent raises ToolException on non-success status."""
    tool = NimbleAgentRunTool(api_key="test_key")
    mock_response = _mock_agent_run_response(
        status="error", warnings=["Something went wrong"]
    )

    with (
        patch.object(tool._sync_client.agent, "run", return_value=mock_response),
        pytest.raises(ToolException, match="status 'error'"),
    ):
        tool._run(agent="bad_agent", params={"url": "https://example.com"})


def test_nimble_agent_run_tool_fatal_status() -> None:
    """Test agent raises ToolException on fatal status."""
    tool = NimbleAgentRunTool(api_key="test_key")
    mock_response = _mock_agent_run_response(status="fatal")

    with (
        patch.object(tool._sync_client.agent, "run", return_value=mock_response),
        pytest.raises(ToolException, match="status 'fatal'"),
    ):
        tool._run(agent="bad_agent", params={})


def test_nimble_agent_run_tool_with_localization() -> None:
    """Test agent run with localization parameter."""
    tool = NimbleAgentRunTool(api_key="test_key")
    mock_response = _mock_agent_run_response()

    with patch.object(
        tool._sync_client.agent, "run", return_value=mock_response
    ) as mock_run:
        tool._run(
            agent="google_search",
            params={"query": "test"},
            localization=True,
        )

    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["localization"] is True


def test_nimble_agent_run_tool_invoke() -> None:
    """Test tool invoke method."""
    tool = NimbleAgentRunTool(api_key="test_key")
    mock_response = _mock_agent_run_response()

    with patch.object(
        tool._sync_client.agent, "run", return_value=mock_response
    ) as mock_run:
        result = tool.invoke({"agent": "google_search", "params": {"query": "test"}})

    assert result is not None
    mock_run.assert_called_once()


async def test_nimble_agent_run_tool_ainvoke() -> None:
    """Test tool async invoke method."""
    tool = NimbleAgentRunTool(api_key="test_key")
    mock_response = _mock_agent_run_response()

    with patch.object(
        tool._async_client.agent, "run", return_value=mock_response
    ) as mock_run:
        result = await tool.ainvoke(
            {"agent": "google_search", "params": {"query": "test"}}
        )

    assert result is not None
    mock_run.assert_awaited_once()


def test_nimble_agent_run_tool_input_validation() -> None:
    """Test NimbleAgentRunToolInput validation."""
    from langchain_nimble.tools.agent_tool import NimbleAgentRunToolInput

    valid_input = NimbleAgentRunToolInput(
        agent="google_search", params={"query": "test"}
    )
    assert valid_input.agent == "google_search"
    assert valid_input.params == {"query": "test"}
    assert valid_input.localization is None
