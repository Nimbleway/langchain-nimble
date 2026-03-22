"""LangChain tools for Nimble Agent API."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_nimble._utilities import _NimbleClientMixin, handle_api_errors

# ───────────────────────────────────────────────────────────────
# nimble_agent_list
# ───────────────────────────────────────────────────────────────


class NimbleAgentListToolInput(BaseModel):
    """Input schema for NimbleAgentListTool."""

    search: str | None = Field(
        default=None,
        description="Search query to filter agents by name or description.",
    )
    managed_by: str | None = Field(
        default=None,
        description="""Filter by who manages the agent.

        Options:
        - "nimble": Official Nimble-managed agents
        - "community": Community-contributed agents
        - "self_managed": Your own custom agents
        """,
    )
    privacy: str | None = Field(
        default=None,
        description="""Filter by privacy level.

        Options:
        - "public": Publicly available agents
        - "private": Your private agents
        - "all": Both public and private
        """,
    )
    limit: int | None = Field(
        default=None,
        description="Maximum number of agents to return.",
    )


class NimbleAgentListTool(_NimbleClientMixin, BaseTool):
    """List available Nimble agent templates.

    Returns agent names, descriptions, and metadata. Use this tool first
    to discover which agents are available before running one.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_agent_list"
    description: str = (
        "List available Nimble agent templates. Returns agent names and "
        "descriptions. Use this first to discover which agents exist, "
        "then use nimble_agent_get to see required parameters."
    )
    args_schema: type[BaseModel] = NimbleAgentListToolInput
    handle_tool_error: bool = True

    def _build_list_kwargs(
        self,
        *,
        search: str | None,
        managed_by: str | None,
        privacy: str | None,
        limit: int | None,
    ) -> dict[str, Any]:
        """Build keyword arguments for SDK agent.list() call."""
        kwargs: dict[str, Any] = {}

        if search is not None:
            kwargs["search"] = search
        if managed_by is not None:
            kwargs["managed_by"] = managed_by
        if privacy is not None:
            kwargs["privacy"] = privacy
        if limit is not None:
            kwargs["limit"] = limit

        return kwargs

    def _run(
        self,
        *,
        search: str | None = None,
        managed_by: str | None = None,
        privacy: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List agents synchronously."""
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        list_kwargs = self._build_list_kwargs(
            search=search,
            managed_by=managed_by,
            privacy=privacy,
            limit=limit,
        )

        with handle_api_errors(operation="agent list"):
            response = self._sync_client.agent.list(**list_kwargs)
            return [agent.model_dump() for agent in response]

    async def _arun(
        self,
        *,
        search: str | None = None,
        managed_by: str | None = None,
        privacy: str | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """List agents asynchronously."""
        if self._async_client is None:
            msg = "Async client not initialized"
            raise RuntimeError(msg)

        list_kwargs = self._build_list_kwargs(
            search=search,
            managed_by=managed_by,
            privacy=privacy,
            limit=limit,
        )

        with handle_api_errors(operation="agent list"):
            response = await self._async_client.agent.list(**list_kwargs)
            return [agent.model_dump() for agent in response]


# ───────────────────────────────────────────────────────────────
# nimble_agent_get
# ───────────────────────────────────────────────────────────────


class NimbleAgentGetToolInput(BaseModel):
    """Input schema for NimbleAgentGetTool."""

    template_name: str = Field(
        description="""The agent template name to get details for.

        Use nimble_agent_list first to discover available agent names,
        then pass one here to see its required parameters and schema.
        Examples: "amazon_pdp", "google_search", "walmart_pdp"
        """,
    )


class NimbleAgentGetTool(_NimbleClientMixin, BaseTool):
    """Get details about a specific Nimble agent template.

    Returns the agent's input parameters (name, type, required, description,
    examples) and output schema. Use this after nimble_agent_list to
    understand what parameters an agent requires before running it.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_agent_get"
    description: str = (
        "Get details about a Nimble agent template including its required "
        "parameters, types, and output schema. Use after nimble_agent_list "
        "to learn what params to pass to nimble_agent_run."
    )
    args_schema: type[BaseModel] = NimbleAgentGetToolInput
    handle_tool_error: bool = True

    def _run(self, template_name: str) -> dict[str, Any]:
        """Get agent details synchronously."""
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        with handle_api_errors(operation="agent get"):
            response = self._sync_client.agent.get(template_name)
            return response.model_dump()

    async def _arun(self, template_name: str) -> dict[str, Any]:
        """Get agent details asynchronously."""
        if self._async_client is None:
            msg = "Async client not initialized"
            raise RuntimeError(msg)

        with handle_api_errors(operation="agent get"):
            response = await self._async_client.agent.get(template_name)
            return response.model_dump()


# ───────────────────────────────────────────────────────────────
# nimble_agent_run
# ───────────────────────────────────────────────────────────────


class NimbleAgentRunToolInput(BaseModel):
    """Input schema for NimbleAgentRunTool."""

    agent: str = Field(
        description="""The agent template name to run.

        Use nimble_agent_list to discover available agents, then
        nimble_agent_get to see required parameters before running.
        Examples: "amazon_pdp", "google_search", "walmart_pdp"
        """,
    )
    params: dict[str, object] = Field(
        description="""Agent-specific parameters.

        Each agent requires different parameters. Use nimble_agent_get
        to discover the exact parameters for your chosen agent.

        Common examples:
        - amazon_pdp: {"asin": "B0..."}
        - google_search: {"query": "search term"}
        - walmart_pdp: {"url": "https://walmart.com/ip/..."}
        """,
    )
    localization: bool | None = Field(
        default=None,
        description="Enable localization for agent results.",
    )


class NimbleAgentRunTool(_NimbleClientMixin, BaseTool):
    """Run a Nimble agent template for structured data collection.

    Agents handle complex workflows like Google search parsing, Amazon
    product extraction, and more. Each agent has unique required parameters.

    Recommended workflow:
    1. nimble_agent_list → discover available agents
    2. nimble_agent_get → see required params for a specific agent
    3. nimble_agent_run → execute with the correct params

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
        locale: Locale for results (default: en).
        country: Country code (default: US).
    """

    name: str = "nimble_agent_run"
    description: str = (
        "Run a Nimble agent template for structured data collection. "
        "Use nimble_agent_list and nimble_agent_get first to discover "
        "agents and their required parameters."
    )
    args_schema: type[BaseModel] = NimbleAgentRunToolInput
    handle_tool_error: bool = True

    def _build_agent_kwargs(
        self,
        agent: str,
        params: dict[str, object],
        *,
        localization: bool | None,
    ) -> dict[str, Any]:
        """Build keyword arguments for SDK agent.run() call."""
        merged_params: dict[str, object] = {"no_html": True, **params}

        kwargs: dict[str, Any] = {
            "agent": agent,
            "params": merged_params,
        }

        if localization is not None:
            kwargs["localization"] = localization

        return kwargs

    def _validate_response(self, response: Any) -> dict[str, Any]:
        """Validate agent response status and return dumped data."""
        if response.status != "success":
            warnings_msg = ""
            if response.warnings:
                warnings_msg = f" Warnings: {response.warnings}"
            msg = (
                f"Agent returned status '{response.status}' "
                f"for task {response.task_id}.{warnings_msg}"
            )
            raise ToolException(msg)
        return response.model_dump()

    def _run(
        self,
        agent: str,
        params: dict[str, object],
        *,
        localization: bool | None = None,
    ) -> dict[str, Any]:
        """Execute agent synchronously."""
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        agent_kwargs = self._build_agent_kwargs(
            agent=agent,
            params=params,
            localization=localization,
        )

        with handle_api_errors(operation="agent run"):
            response = self._sync_client.agent.run(**agent_kwargs)
            return self._validate_response(response)

    async def _arun(
        self,
        agent: str,
        params: dict[str, object],
        *,
        localization: bool | None = None,
    ) -> dict[str, Any]:
        """Execute agent asynchronously."""
        if self._async_client is None:
            msg = "Async client not initialized"
            raise RuntimeError(msg)

        agent_kwargs = self._build_agent_kwargs(
            agent=agent,
            params=params,
            localization=localization,
        )

        with handle_api_errors(operation="agent run"):
            response = await self._async_client.agent.run(**agent_kwargs)
            return self._validate_response(response)
