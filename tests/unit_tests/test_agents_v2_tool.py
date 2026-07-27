"""Unit tests for Nimble Agent API V2 tools."""

from unittest.mock import MagicMock, patch

from langchain_nimble import (
    NimbleAgentCreateTool,
    NimbleAgentRunResultTool,
    NimbleAgentRunStartTool,
    NimbleAgentRunStatusTool,
    NimbleAgentsListTool,
    NimbleAgentTemplatesListTool,
)


def _mock_items_response(items: list[dict[str, object]]) -> MagicMock:
    """Create a mock list response with .items."""
    response = MagicMock()
    mocks = []
    for item in items:
        mock_item = MagicMock()
        mock_item.model_dump.return_value = item
        mocks.append(mock_item)
    response.items = mocks
    return response


def test_agents_list_run() -> None:
    """Test synchronous agents list."""
    tool = NimbleAgentsListTool(api_key="test_key")
    mock_response = _mock_items_response(
        [{"id": "wsa_123", "agent_name": "research"}],
    )

    with patch.object(
        tool._sync_client.agents,
        "list",
        return_value=mock_response,
    ) as mock_list:
        result = tool._run(limit=5)

    assert result[0]["id"] == "wsa_123"
    mock_list.assert_called_once_with(limit=5)


async def test_agents_list_arun() -> None:
    """Test asynchronous agents list."""
    tool = NimbleAgentsListTool(api_key="test_key")
    mock_response = _mock_items_response([{"id": "wsa_123"}])

    with patch.object(
        tool._async_client.agents,
        "list",
        return_value=mock_response,
    ) as mock_list:
        result = await tool._arun(workspace_id="ws_1")

    assert result[0]["id"] == "wsa_123"
    mock_list.assert_awaited_once_with(workspace_id="ws_1")


def test_agent_templates_list_run() -> None:
    """Test synchronous agent templates list."""
    tool = NimbleAgentTemplatesListTool(api_key="test_key")
    mock_response = _mock_items_response([{"name": "deep_research"}])

    with patch.object(
        tool._sync_client.agents.templates,
        "list",
        return_value=mock_response,
    ) as mock_list:
        result = tool._run()

    assert result[0]["name"] == "deep_research"
    mock_list.assert_called_once()


def test_agent_create_run() -> None:
    """Test synchronous agent create."""
    tool = NimbleAgentCreateTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "id": "wsa_new",
        "agent_name": "my_agent",
    }

    with patch.object(
        tool._sync_client.agents,
        "create",
        return_value=mock_response,
    ) as mock_create:
        result = tool._run(template="deep_research", display_name="My Agent")

    assert result["id"] == "wsa_new"
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["template"] == "deep_research"
    assert call_kwargs["display_name"] == "My Agent"


async def test_agent_create_arun() -> None:
    """Test asynchronous agent create."""
    tool = NimbleAgentCreateTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"id": "wsa_new"}

    with patch.object(
        tool._async_client.agents,
        "create",
        return_value=mock_response,
    ) as mock_create:
        result = await tool._arun(effort="high", use_case="research")

    assert result["id"] == "wsa_new"
    mock_create.assert_awaited_once()


def test_agent_run_start() -> None:
    """Test synchronous agent run start returns immediately."""
    tool = NimbleAgentRunStartTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "id": "task_run_abc",
        "status": "queued",
        "web_search_agent_id": "wsa_123",
    }

    with patch.object(
        tool._sync_client.agents.runs,
        "create",
        return_value=mock_response,
    ) as mock_create:
        result = tool._run(
            agent_id="wsa_123",
            input="Research AI agents",
            effort="medium",
        )

    assert result["id"] == "task_run_abc"
    assert result["status"] == "queued"
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["agent_id"] == "wsa_123"
    assert call_kwargs["input"] == "Research AI agents"
    assert call_kwargs["effort"] == "medium"


async def test_agent_run_start_arun() -> None:
    """Test asynchronous agent run start."""
    tool = NimbleAgentRunStartTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"id": "task_run_abc", "status": "running"}

    with patch.object(
        tool._async_client.agents.runs,
        "create",
        return_value=mock_response,
    ) as mock_create:
        result = await tool._arun(agent_id="wsa_123", input="hello")

    assert result["id"] == "task_run_abc"
    mock_create.assert_awaited_once()


def test_agent_run_status() -> None:
    """Test synchronous agent run status (non-terminal is ok)."""
    tool = NimbleAgentRunStatusTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "id": "task_run_abc",
        "status": "running",
    }

    with patch.object(
        tool._sync_client.agents.runs,
        "get",
        return_value=mock_response,
    ) as mock_get:
        result = tool._run(agent_id="wsa_123", run_id="task_run_abc")

    assert result["status"] == "running"
    mock_get.assert_called_once_with("task_run_abc", agent_id="wsa_123")


async def test_agent_run_status_arun() -> None:
    """Test asynchronous agent run status."""
    tool = NimbleAgentRunStatusTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"id": "task_run_abc", "status": "queued"}

    with patch.object(
        tool._async_client.agents.runs,
        "get",
        return_value=mock_response,
    ) as mock_get:
        result = await tool._arun(agent_id="wsa_123", run_id="task_run_abc")

    assert result["status"] == "queued"
    mock_get.assert_awaited_once()


def test_agent_run_result() -> None:
    """Test synchronous agent run result."""
    tool = NimbleAgentRunResultTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "result": "Research complete",
        "citations": [{"url": "https://example.com"}],
    }

    with patch.object(
        tool._sync_client.agents.runs,
        "result",
        return_value=mock_response,
    ) as mock_result:
        result = tool._run(agent_id="wsa_123", run_id="task_run_abc")

    assert result["result"] == "Research complete"
    mock_result.assert_called_once_with("task_run_abc", agent_id="wsa_123")


async def test_agent_run_result_arun() -> None:
    """Test asynchronous agent run result."""
    tool = NimbleAgentRunResultTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"result": "done"}

    with patch.object(
        tool._async_client.agents.runs,
        "result",
        return_value=mock_response,
    ) as mock_result:
        result = await tool._arun(agent_id="wsa_123", run_id="task_run_abc")

    assert result["result"] == "done"
    mock_result.assert_awaited_once()


def test_tool_names() -> None:
    """Test Agent API V2 tools have expected names."""
    assert NimbleAgentsListTool(api_key="k").name == "nimble_agents_list"
    templates = NimbleAgentTemplatesListTool(api_key="k")
    assert templates.name == "nimble_agent_templates_list"
    assert NimbleAgentCreateTool(api_key="k").name == "nimble_agent_create"
    assert NimbleAgentRunStartTool(api_key="k").name == "nimble_agent_run_start"
    assert NimbleAgentRunStatusTool(api_key="k").name == "nimble_agent_run_status"
    assert NimbleAgentRunResultTool(api_key="k").name == "nimble_agent_run_result"
