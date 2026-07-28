"""Unit tests for Nimble Agent API V2 tools."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException

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


def test_agent_run_start_mode2() -> None:
    """Test Mode 2 start uses agents.runs.create(agent_id, ...)."""
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
            skill="Focus on primary sources",
            use_case="research",
        )

    assert result["id"] == "task_run_abc"
    assert result["status"] == "queued"
    assert mock_create.call_args.args == ("wsa_123",)
    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["input"] == "Research AI agents"
    assert call_kwargs["effort"] == "medium"
    assert call_kwargs["extra_body"] == {
        "use_case": "research",
        "skill": "Focus on primary sources",
    }


def test_agent_run_start_mode1_agent_name() -> None:
    """Test Mode 1 start uses agents.run with agent_name in extra_body."""
    tool = NimbleAgentRunStartTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "id": "task_run_abc",
        "status": "queued",
        "web_search_agent_id": "wsa_new",
    }

    with patch.object(
        tool._sync_client.agents,
        "run",
        return_value=mock_response,
    ) as mock_run:
        result = tool._run(
            agent_name="integrations_research_bot",
            input="Summarize Agent API v2",
            use_case="research",
            effort="medium",
            skill="Integrator-focused docs",
            sources={"prioritize": "official docs", "avoid": "spam blogs"},
        )

    assert result["web_search_agent_id"] == "wsa_new"
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["input"] == "Summarize Agent API v2"
    assert call_kwargs["effort"] == "medium"
    assert call_kwargs["sources"]["prioritize"] == "official docs"
    assert call_kwargs["extra_body"] == {
        "agent_name": "integrations_research_bot",
        "use_case": "research",
        "skill": "Integrator-focused docs",
    }


def test_agent_run_start_mode3_anonymous() -> None:
    """Test Mode 3 start omits agent_id/agent_name."""
    tool = NimbleAgentRunStartTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "id": "task_run_abc",
        "web_search_agent_id": "wsa_anon",
    }

    with patch.object(
        tool._sync_client.agents,
        "run",
        return_value=mock_response,
    ) as mock_run:
        result = tool._run(input="Quick one-shot research")

    assert result["web_search_agent_id"] == "wsa_anon"
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["input"] == "Quick one-shot research"
    assert "extra_body" not in call_kwargs


async def test_agent_run_start_arun() -> None:
    """Test asynchronous Mode 2 agent run start."""
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


def test_agent_create_includes_skill_and_sources() -> None:
    """Test create maps skill/sources/output_schema to SDK kwargs."""
    tool = NimbleAgentCreateTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"id": "wsa_new"}

    with patch.object(
        tool._sync_client.agents,
        "create",
        return_value=mock_response,
    ) as mock_create:
        tool._run(
            agent_name="enrich_bot",
            use_case="enrichment",
            skill="Company firmographics",
            sources={
                "allow": [
                    {"title": "Filings", "domains": ["sec.gov"], "order": 0},
                ],
            },
            output_schema={"type": "object"},
        )

    call_kwargs = mock_create.call_args.kwargs
    assert call_kwargs["skill"] == "Company firmographics"
    assert call_kwargs["use_case"] == "enrichment"
    assert call_kwargs["output_schema"] == {"type": "object"}
    assert call_kwargs["sources"]["allow"][0]["domains"] == ["sec.gov"]


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


def test_agent_run_result_failed_raises_tool_exception() -> None:
    """Test failed run_result payloads raise ToolException."""
    tool = NimbleAgentRunResultTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {
        "error": {"message": "budget exceeded", "code": "failed"},
        "run": {"id": "task_run_abc", "status": "failed"},
    }

    with (
        patch.object(
            tool._sync_client.agents.runs,
            "result",
            return_value=mock_response,
        ),
        pytest.raises(ToolException, match="Agent run failed"),
    ):
        tool._run(agent_id="wsa_123", run_id="task_run_abc")


def test_missing_client_raises_contextual_tool_exception() -> None:
    """Test missing sync client raises ToolException with tool name."""
    tool = NimbleAgentRunStartTool(api_key="test_key")
    tool._sync_client = None

    with pytest.raises(
        ToolException,
        match="nimble_web_search_agent_run_start: sync client not initialized",
    ):
        tool._run(agent_id="wsa_123", input="hello")


def test_tool_names() -> None:
    """Test Agent API V2 tools have expected names."""
    assert NimbleAgentsListTool(api_key="k").name == "nimble_web_search_agents_list"
    templates = NimbleAgentTemplatesListTool(api_key="k")
    assert templates.name == "nimble_web_search_agent_templates_list"
    assert NimbleAgentCreateTool(api_key="k").name == "nimble_web_search_agent_create"
    assert (
        NimbleAgentRunStartTool(api_key="k").name == "nimble_web_search_agent_run_start"
    )
    assert (
        NimbleAgentRunStatusTool(api_key="k").name
        == "nimble_web_search_agent_run_status"
    )
    assert (
        NimbleAgentRunResultTool(api_key="k").name
        == "nimble_web_search_agent_run_result"
    )
