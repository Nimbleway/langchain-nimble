"""Integration smokes for Extract Templates and Agent API V2.

Requires NIMBLE_API_KEY environment variable.

Billable start/run tests are marked ``expensive`` — run with::

    pytest tests/integration_tests/test_extract_templates_and_agents.py -m expensive
"""

from __future__ import annotations

import os
import time
from typing import Any

import pytest
from langchain_core.tools import ToolException

from langchain_nimble import (
    NimbleAgentRunResultTool,
    NimbleAgentRunStartTool,
    NimbleAgentRunStatusTool,
    NimbleAgentsListTool,
    NimbleExtractTemplateGetTool,
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
_TERMINAL_STATUSES = {"completed", "failed", "cancelled", "canceled"}


@pytest.fixture
def api_key() -> str:
    """Get API key from environment or skip test."""
    key = os.environ.get("NIMBLE_API_KEY")
    if not key:
        pytest.skip("NIMBLE_API_KEY not set")
    return key


def _candidate_params(published: dict[str, Any]) -> dict[str, object] | None:
    """Build params from published_version samples or input_schema examples.

    Args:
        published: Extract template published_version payload.

    Returns:
        Params dict, or ``None`` if none can be inferred.
    """
    samples = published.get("samples") or []
    if isinstance(samples, list):
        for sample in samples:
            if not isinstance(sample, dict):
                continue
            sample_input = sample.get("input")
            if isinstance(sample_input, dict) and sample_input:
                return dict(sample_input)

    schema = published.get("input_schema") or {}
    if not isinstance(schema, dict):
        return None
    props = schema.get("properties") or {}
    if not isinstance(props, dict):
        return None
    params: dict[str, object] = {}
    for key, prop in props.items():
        if not isinstance(prop, dict):
            continue
        examples = prop.get("examples") or []
        if isinstance(examples, list) and examples:
            params[key] = examples[0]
    required = schema.get("required") or []
    if isinstance(required, list) and required:
        if all(r in params for r in required if isinstance(r, str)):
            return params
        return None
    return params or None


def _first_runnable_template(
    api_key: str,
    *,
    limit: int = 80,
) -> tuple[str, dict[str, object], dict[str, Any]]:
    """Pick an extract template that successfully runs with example params.

    Some listed templates return 404 on run for this account; probe until one
    succeeds.

    Args:
        api_key: Nimble API key.
        limit: Max templates to scan from list.

    Returns:
        Tuple of template name, params, and successful run payload.

    Raises:
        pytest.skip.Exception: If no runnable template is found.
    """
    list_tool = NimbleExtractTemplateListTool(api_key=api_key)
    get_tool = NimbleExtractTemplateGetTool(api_key=api_key)
    run_tool = NimbleExtractTemplateRunTool(api_key=api_key)
    templates = list_tool._run(limit=limit)
    if not templates:
        pytest.skip("No extract templates available for this account")

    # Prefer URL PDP templates — more likely executable on this account.
    ordered = sorted(
        templates,
        key=lambda item: (
            0
            if isinstance(item, dict)
            and isinstance(item.get("name"), str)
            and str(item["name"]).endswith("_pdp")
            else 1
        ),
    )

    for item in ordered:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name:
            continue
        details = get_tool._run(template_name=name)
        published = details.get("published_version") or {}
        if not isinstance(published, dict):
            continue
        params = _candidate_params(published)
        if not params:
            continue
        try:
            result = run_tool._run(template=name, params=params)
        except ToolException as exc:
            if "404" in str(exc) or "Template not found" in str(exc):
                continue
            raise
        if isinstance(result, dict) and result.get("status") == "success":
            return name, params, result

    pytest.skip("No executable extract template found for this account")


def _poll_until_terminal(
    *,
    api_key: str,
    agent_id: str,
    run_id: str,
    timeout_s: float = 300.0,
    interval_s: float = 5.0,
) -> dict[str, Any]:
    """Poll run status until terminal or timeout.

    Args:
        api_key: Nimble API key.
        agent_id: Web Search Agent id.
        run_id: Run id.
        timeout_s: Max seconds to wait.
        interval_s: Seconds between polls.

    Returns:
        Final status payload.

    Raises:
        AssertionError: If the run does not reach a terminal status in time.
    """
    status_tool = NimbleAgentRunStatusTool(api_key=api_key)
    deadline = time.monotonic() + timeout_s
    last: dict[str, Any] = {}
    while time.monotonic() < deadline:
        status = status_tool.invoke({"agent_id": agent_id, "run_id": run_id})
        assert isinstance(status, dict)
        last = status
        current = status.get("status")
        assert current in _AGENT_RUN_STATUSES
        if current in _TERMINAL_STATUSES:
            return status
        time.sleep(interval_s)
    msg = f"Run {run_id} did not finish within {timeout_s}s; last={last}"
    raise AssertionError(msg)


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
    """Live smoke: run a discoverable extract template from this account."""
    template, params, result = _first_runnable_template(api_key)

    assert template
    assert params
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
    """Live smoke: Mode 2 start + status when an agent already exists."""
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


@pytest.mark.expensive
def test_agent_run_start_mode1_live(api_key: str) -> None:
    """Live smoke: Mode 1 agent_name create-or-reuse + status."""
    start_tool = NimbleAgentRunStartTool(api_key=api_key)
    started = start_tool.invoke(
        {
            "agent_name": "langchain_nimble_mode1_smoke",
            "use_case": "research",
            "effort": "low",
            "skill": "One short sentence; prefer official docs",
            "input": "Say hello in one short sentence.",
        }
    )

    assert isinstance(started, dict)
    run_id = started.get("id")
    agent_id = started.get("web_search_agent_id")
    assert run_id, f"Expected run id: {started}"
    assert agent_id, f"Expected web_search_agent_id: {started}"
    assert str(agent_id).startswith("wsa_")

    status_tool = NimbleAgentRunStatusTool(api_key=api_key)
    status = status_tool.invoke({"agent_id": agent_id, "run_id": run_id})
    assert status.get("status") in _AGENT_RUN_STATUSES


@pytest.mark.expensive
def test_agent_research_e2e_mode1_live(api_key: str) -> None:
    """Live E2E: Mode 1 research through terminal status + result."""
    start_tool = NimbleAgentRunStartTool(api_key=api_key)
    started = start_tool.invoke(
        {
            "agent_name": "langchain_nimble_research_e2e",
            "use_case": "research",
            "effort": "low",
            "skill": "Reply in one short sentence.",
            "input": "What is LangChain in one sentence?",
        }
    )
    run_id = started["id"]
    agent_id = started["web_search_agent_id"]
    assert str(run_id).startswith("task_run_")
    assert str(agent_id).startswith("wsa_")

    final = _poll_until_terminal(
        api_key=api_key,
        agent_id=agent_id,
        run_id=run_id,
        timeout_s=300.0,
    )
    assert final["status"] in _TERMINAL_STATUSES

    result_tool = NimbleAgentRunResultTool(api_key=api_key)
    if final["status"] == "completed":
        result = result_tool.invoke({"agent_id": agent_id, "run_id": run_id})
        assert isinstance(result, dict)
        assert "output" in result
        output = result["output"]
        assert isinstance(output, dict)
        assert output.get("type") == "text"
        assert output.get("content")
        trust = output.get("trust")
        assert trust is None or isinstance(trust, dict)
    else:
        with pytest.raises(ToolException, match="Agent run failed"):
            result_tool.invoke({"agent_id": agent_id, "run_id": run_id})


@pytest.mark.expensive
def test_agent_enrichment_e2e_mode1_live(api_key: str) -> None:
    """Live E2E: Mode 1 enrichment with input_data through result."""
    start_tool = NimbleAgentRunStartTool(api_key=api_key)
    schema = {
        "type": "object",
        "properties": {
            "domain": {"type": "string"},
            "company_name": {"type": "string"},
        },
        "required": ["domain", "company_name"],
    }
    started = start_tool.invoke(
        {
            "agent_name": "langchain_nimble_enrichment_e2e",
            "use_case": "enrichment",
            "effort": "low",
            "skill": "Fill missing company_name from the domain only.",
            "output_schema": schema,
            "input_data": [{"domain": "langchain.com"}],
            "input": "Enrich the company row.",
        }
    )
    run_id = started["id"]
    agent_id = started["web_search_agent_id"]

    final = _poll_until_terminal(
        api_key=api_key,
        agent_id=agent_id,
        run_id=run_id,
        timeout_s=300.0,
    )
    assert final["status"] in _TERMINAL_STATUSES

    result_tool = NimbleAgentRunResultTool(api_key=api_key)
    if final["status"] != "completed":
        pytest.skip(f"Enrichment run ended as {final['status']}; start/status OK")

    result = result_tool.invoke({"agent_id": agent_id, "run_id": run_id})
    assert result.get("output", {}).get("type") == "json"
    assert result["output"].get("content") is not None


@pytest.mark.expensive
def test_agent_dataset_building_e2e_mode1_live(api_key: str) -> None:
    """Live E2E: Mode 1 dataset_building through result.

    API requires effort ``high`` or higher for ``dataset_building``.
    """
    start_tool = NimbleAgentRunStartTool(api_key=api_key)
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "url": {"type": "string"},
                    },
                    "required": ["name"],
                },
            }
        },
        "required": ["items"],
    }
    started = start_tool._run(
        input="List two well-known open-source LLM frameworks.",
        agent_name="langchain_nimble_dataset_e2e",
        use_case="dataset_building",
        effort="high",
        skill="Return at most two items.",
        output_schema=schema,
    )
    run_id = started["id"]
    agent_id = started["web_search_agent_id"]

    final = _poll_until_terminal(
        api_key=api_key,
        agent_id=agent_id,
        run_id=run_id,
        timeout_s=900.0,
        interval_s=10.0,
    )
    assert final["status"] in _TERMINAL_STATUSES

    result_tool = NimbleAgentRunResultTool(api_key=api_key)
    if final["status"] != "completed":
        pytest.skip(f"Dataset run ended as {final['status']}; start/status OK")

    result = result_tool.invoke({"agent_id": agent_id, "run_id": run_id})
    assert result.get("output", {}).get("type") == "json"
    assert result["output"].get("content") is not None


@pytest.mark.expensive
def test_agent_use_case_lock_422_live(api_key: str) -> None:
    """Live: mismatched use_case against an existing agent returns 422."""
    start_tool = NimbleAgentRunStartTool(api_key=api_key)
    # Ensure research agent exists
    start_tool._run(
        input="ping",
        agent_name="langchain_nimble_usecase_lock",
        use_case="research",
        effort="low",
    )

    with pytest.raises(ToolException, match="422") as exc_info:
        start_tool._run(
            input="ping again",
            agent_name="langchain_nimble_usecase_lock",
            use_case="enrichment",
            effort="low",
        )
    assert "422" in str(exc_info.value)


@pytest.mark.expensive
def test_client_source_header_live(api_key: str) -> None:
    """Live: captured request includes X-Client-Source langchain-nimble."""
    # Use package tool client so attribution matches production path
    tool = NimbleAgentsListTool(api_key=api_key)
    assert tool._sync_client is not None
    assert tool._sync_client.client_source == "langchain-nimble"

    captured: dict[str, str] = {}

    def _capture_request(request: Any) -> None:
        captured["x-client-source"] = request.headers.get("x-client-source", "")

    http = tool._sync_client._client
    http.event_hooks.setdefault("request", []).append(_capture_request)
    tool._run(limit=1)
    assert captured.get("x-client-source") == "langchain-nimble"
