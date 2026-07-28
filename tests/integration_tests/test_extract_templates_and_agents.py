"""Integration smokes for Extract Templates and Agent API V2.

Requires NIMBLE_API_KEY environment variable.

Billable start/run tests are marked ``expensive`` — run with::

    pytest tests/integration_tests/ -m expensive
"""

from __future__ import annotations

import os

import pytest

from langchain_nimble import (
    NimbleAgentRunStartTool,
    NimbleAgentRunStatusTool,
    NimbleAgentsListTool,
    NimbleExtractTemplateListTool,
    NimbleExtractTemplateRunTool,
)

_AGENT_RUN_STATUSES = {
    "queued",
    "running",
    "completed",
    "failed",
    "cancelled",
    "canceled",
    "success",
    "error",
}


@pytest.fixture
def api_key() -> str:
    """Get API key from environment or skip test."""
    key = os.environ.get("NIMBLE_API_KEY")
    if not key:
        pytest.skip("NIMBLE_API_KEY not set")
    return key


def test_extract_template_list_live(api_key: str) -> None:
    """Live smoke: list extract templates."""
    tool = NimbleExtractTemplateListTool(api_key=api_key)
    result = tool.invoke({"limit": 5})

    assert isinstance(result, list)
    if result:
        assert isinstance(result[0], dict)
        assert "name" in result[0] or "id" in result[0]


@pytest.mark.expensive
def test_extract_template_run_live(api_key: str) -> None:
    """Live smoke: run a known extract template when available."""
    list_tool = NimbleExtractTemplateListTool(api_key=api_key)
    templates = list_tool.invoke({"limit": 20})
    if not templates:
        pytest.skip("No extract templates available for this account")

    names = {t.get("name") for t in templates if isinstance(t, dict)}
    if "google_search" not in names:
        pytest.skip("google_search extract template not available")

    run_tool = NimbleExtractTemplateRunTool(api_key=api_key)
    result = run_tool.invoke(
        {
            "template": "google_search",
            "params": {"query": "langchain nimble"},
        }
    )

    assert isinstance(result, dict)
    assert result.get("status") == "success"
    assert result.get("task_id")
    assert result.get("data") is not None


def test_agents_list_live(api_key: str) -> None:
    """Live smoke: list Agent API V2 agents."""
    tool = NimbleAgentsListTool(api_key=api_key)
    result = tool.invoke({"limit": 5})

    assert isinstance(result, list)
    if result:
        assert isinstance(result[0], dict)
        assert "id" in result[0]


@pytest.mark.expensive
def test_agent_run_start_and_status_live(api_key: str) -> None:
    """Live smoke: start a run and fetch status (may be non-terminal)."""
    list_tool = NimbleAgentsListTool(api_key=api_key)
    agents = list_tool.invoke({"limit": 5})
    if not agents:
        pytest.skip("No Web Search Agents available for this account")

    agent_id = agents[0].get("id")
    if not agent_id:
        pytest.skip("Agent list item missing id")

    start_tool = NimbleAgentRunStartTool(api_key=api_key)
    started = start_tool.invoke(
        {
            "agent_id": agent_id,
            "input": "Say hello in one short sentence.",
            "effort": "low",
        }
    )

    assert isinstance(started, dict)
    run_id = started.get("id")
    assert run_id, f"Expected run id in start response: {started}"
    wsa_id = started.get("web_search_agent_id")
    if wsa_id is not None:
        assert wsa_id == agent_id

    status_tool = NimbleAgentRunStatusTool(api_key=api_key)
    status = status_tool.invoke({"agent_id": agent_id, "run_id": run_id})

    assert isinstance(status, dict)
    assert status.get("status") in _AGENT_RUN_STATUSES
