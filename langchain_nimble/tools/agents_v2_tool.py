"""LangChain tools for Nimble Agent API V2 (Web Search Agents)."""

from __future__ import annotations

from typing import Any, Literal

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_nimble._utilities import (
    _NimbleClientMixin,
    handle_api_errors,
    require_initialized_client,
)

AgentEffort = Literal["low", "medium", "high", "x-high", "max"]
AgentUseCase = Literal["research", "enrichment", "dataset_building"]


# ───────────────────────────────────────────────────────────────
# nimble_web_search_agents_list
# ───────────────────────────────────────────────────────────────


class NimbleAgentsListToolInput(BaseModel):
    """Input schema for NimbleAgentsListTool.

    Accepts optional pagination and workspace filters for listing agents.
    """

    limit: int | None = Field(
        default=None,
        description="Maximum number of Web Search Agents to return.",
    )
    offset: int | None = Field(
        default=None,
        description="Pagination offset for agent listing.",
    )
    workspace_id: str | None = Field(
        default=None,
        description="Optional workspace id to filter agents.",
    )


class NimbleAgentsListTool(_NimbleClientMixin, BaseTool):
    """List Nimble Web Search Agents (Agent API V2).

    Use this to discover research agents (``wsa_…`` ids) before starting a run.
    This is distinct from Extract Templates (structured site scraping).

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_web_search_agents_list"
    description: str = (
        "List Nimble Web Search Agents (Agent API V2 research agents). "
        "Returns agent ids (often wsa_…) and metadata. Use before "
        "nimble_web_search_agent_run_start. This is NOT Extract Templates — use "
        "nimble_extract_template_list for structured site scraping."
    )
    args_schema: type[BaseModel] = NimbleAgentsListToolInput
    handle_tool_error: bool = True

    def _build_list_kwargs(
        self,
        *,
        limit: int | None,
        offset: int | None,
        workspace_id: str | None,
    ) -> dict[str, Any]:
        """Build keyword arguments for agents.list().

        Args:
            limit: Maximum number of agents to return.
            offset: Pagination offset.
            workspace_id: Optional workspace filter.

        Returns:
            Keyword arguments accepted by ``agents.list``.
        """
        kwargs: dict[str, Any] = {}
        if limit is not None:
            kwargs["limit"] = limit
        if offset is not None:
            kwargs["offset"] = offset
        if workspace_id is not None:
            kwargs["workspace_id"] = workspace_id
        return kwargs

    def _run(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List agents synchronously.

        Args:
            limit: Maximum number of agents to return.
            offset: Pagination offset.
            workspace_id: Optional workspace filter.

        Returns:
            List of agent records as dictionaries.
        """
        require_initialized_client(self.name, self._sync_client, sync=True)

        list_kwargs = self._build_list_kwargs(
            limit=limit,
            offset=offset,
            workspace_id=workspace_id,
        )

        with handle_api_errors(operation="agents list"):
            response = self._sync_client.agents.list(**list_kwargs)  # type: ignore[union-attr]
            return [item.model_dump(mode="json") for item in response.items]

    async def _arun(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
        workspace_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """List agents asynchronously.

        Args:
            limit: Maximum number of agents to return.
            offset: Pagination offset.
            workspace_id: Optional workspace filter.

        Returns:
            List of agent records as dictionaries.
        """
        require_initialized_client(self.name, self._async_client, sync=False)

        list_kwargs = self._build_list_kwargs(
            limit=limit,
            offset=offset,
            workspace_id=workspace_id,
        )

        with handle_api_errors(operation="agents list"):
            response = await self._async_client.agents.list(**list_kwargs)  # type: ignore[union-attr]
            return [item.model_dump(mode="json") for item in response.items]


# ───────────────────────────────────────────────────────────────
# nimble_web_search_agent_templates_list
# ───────────────────────────────────────────────────────────────


class NimbleAgentTemplatesListToolInput(BaseModel):
    """Input schema for NimbleAgentTemplatesListTool.

    Accepts optional pagination for listing Web Search Agent templates.
    """

    limit: int | None = Field(
        default=None,
        description="Maximum number of agent templates to return.",
    )
    offset: int | None = Field(
        default=None,
        description="Pagination offset for agent template listing.",
    )


class NimbleAgentTemplatesListTool(_NimbleClientMixin, BaseTool):
    """List Nimble Web Search Agent templates (Agent API V2).

    Templates can be used with ``nimble_web_search_agent_create`` to instantiate
    an agent.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_web_search_agent_templates_list"
    description: str = (
        "List Nimble Web Search Agent templates (Agent API V2). Use these "
        "template names with nimble_web_search_agent_create to create a "
        "research agent. This is distinct from Extract Templates "
        "(nimble_extract_template_list)."
    )
    args_schema: type[BaseModel] = NimbleAgentTemplatesListToolInput
    handle_tool_error: bool = True

    def _build_list_kwargs(
        self,
        *,
        limit: int | None,
        offset: int | None,
    ) -> dict[str, Any]:
        """Build keyword arguments for agents.templates.list().

        Args:
            limit: Maximum number of templates to return.
            offset: Pagination offset.

        Returns:
            Keyword arguments accepted by ``agents.templates.list``.
        """
        kwargs: dict[str, Any] = {}
        if limit is not None:
            kwargs["limit"] = limit
        if offset is not None:
            kwargs["offset"] = offset
        return kwargs

    def _run(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        """List agent templates synchronously.

        Args:
            limit: Maximum number of templates to return.
            offset: Pagination offset.

        Returns:
            List of template records as dictionaries.
        """
        require_initialized_client(self.name, self._sync_client, sync=True)

        list_kwargs = self._build_list_kwargs(limit=limit, offset=offset)

        with handle_api_errors(operation="agent templates list"):
            response = self._sync_client.agents.templates.list(**list_kwargs)  # type: ignore[union-attr]
            return [item.model_dump(mode="json") for item in response.items]

    async def _arun(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        """List agent templates asynchronously.

        Args:
            limit: Maximum number of templates to return.
            offset: Pagination offset.

        Returns:
            List of template records as dictionaries.
        """
        require_initialized_client(self.name, self._async_client, sync=False)

        list_kwargs = self._build_list_kwargs(limit=limit, offset=offset)

        with handle_api_errors(operation="agent templates list"):
            response = await self._async_client.agents.templates.list(**list_kwargs)  # type: ignore[union-attr]
            return [item.model_dump(mode="json") for item in response.items]


# ───────────────────────────────────────────────────────────────
# nimble_web_search_agent_create
# ───────────────────────────────────────────────────────────────


class NimbleAgentCreateToolInput(BaseModel):
    """Input schema for NimbleAgentCreateTool.

    Optional fields for creating a Web Search Agent from a template or custom
    configuration.
    """

    template: str | None = Field(
        default=None,
        description="Optional agent template name to create from.",
    )
    agent_name: str | None = Field(
        default=None,
        description="Optional unique agent name.",
    )
    display_name: str | None = Field(
        default=None,
        description="Optional human-readable display name.",
    )
    description: str | None = Field(
        default=None,
        description="Optional agent description.",
    )
    effort: AgentEffort | None = Field(
        default=None,
        description="Default effort level: low, medium, high, x-high, or max.",
    )
    use_case: AgentUseCase | None = Field(
        default=None,
        description="Optional use case: research, enrichment, or dataset_building.",
    )
    goals: list[str] | None = Field(
        default=None,
        description="Optional list of agent goals.",
    )


class NimbleAgentCreateTool(_NimbleClientMixin, BaseTool):
    """Create a Nimble Web Search Agent (Agent API V2).

    Prefer creating from a template listed by
    ``nimble_web_search_agent_templates_list``. Returns an agent id (often
    ``wsa_…``) for later run start/status/result calls.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_web_search_agent_create"
    description: str = (
        "Create a Nimble Web Search Agent (Agent API V2). Optionally pass a "
        "template from nimble_web_search_agent_templates_list. Returns an "
        "agent id (often wsa_…) for nimble_web_search_agent_run_start. This "
        "is research agent creation, not Extract Templates."
    )
    args_schema: type[BaseModel] = NimbleAgentCreateToolInput
    handle_tool_error: bool = True

    def _build_create_kwargs(
        self,
        *,
        template: str | None,
        agent_name: str | None,
        display_name: str | None,
        description: str | None,
        effort: AgentEffort | None,
        use_case: AgentUseCase | None,
        goals: list[str] | None,
    ) -> dict[str, Any]:
        """Build keyword arguments for agents.create().

        Args:
            template: Optional template name to create from.
            agent_name: Optional unique agent name.
            display_name: Optional display name.
            description: Optional description.
            effort: Optional default effort level.
            use_case: Optional use case.
            goals: Optional list of goals.

        Returns:
            Keyword arguments accepted by ``agents.create``.
        """
        kwargs: dict[str, Any] = {}
        if template is not None:
            kwargs["template"] = template
        if agent_name is not None:
            kwargs["agent_name"] = agent_name
        if display_name is not None:
            kwargs["display_name"] = display_name
        if description is not None:
            kwargs["description"] = description
        if effort is not None:
            kwargs["effort"] = effort
        if use_case is not None:
            kwargs["use_case"] = use_case
        if goals is not None:
            kwargs["goals"] = goals
        return kwargs

    def _run(
        self,
        *,
        template: str | None = None,
        agent_name: str | None = None,
        display_name: str | None = None,
        description: str | None = None,
        effort: AgentEffort | None = None,
        use_case: AgentUseCase | None = None,
        goals: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create an agent synchronously.

        Args:
            template: Optional template name to create from.
            agent_name: Optional unique agent name.
            display_name: Optional display name.
            description: Optional description.
            effort: Optional default effort level.
            use_case: Optional use case.
            goals: Optional list of goals.

        Returns:
            Created agent record as a dictionary.
        """
        require_initialized_client(self.name, self._sync_client, sync=True)

        create_kwargs = self._build_create_kwargs(
            template=template,
            agent_name=agent_name,
            display_name=display_name,
            description=description,
            effort=effort,
            use_case=use_case,
            goals=goals,
        )

        with handle_api_errors(operation="agent create"):
            response = self._sync_client.agents.create(**create_kwargs)  # type: ignore[union-attr]
            return response.model_dump(mode="json")

    async def _arun(
        self,
        *,
        template: str | None = None,
        agent_name: str | None = None,
        display_name: str | None = None,
        description: str | None = None,
        effort: AgentEffort | None = None,
        use_case: AgentUseCase | None = None,
        goals: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create an agent asynchronously.

        Args:
            template: Optional template name to create from.
            agent_name: Optional unique agent name.
            display_name: Optional display name.
            description: Optional description.
            effort: Optional default effort level.
            use_case: Optional use case.
            goals: Optional list of goals.

        Returns:
            Created agent record as a dictionary.
        """
        require_initialized_client(self.name, self._async_client, sync=False)

        create_kwargs = self._build_create_kwargs(
            template=template,
            agent_name=agent_name,
            display_name=display_name,
            description=description,
            effort=effort,
            use_case=use_case,
            goals=goals,
        )

        with handle_api_errors(operation="agent create"):
            response = await self._async_client.agents.create(**create_kwargs)  # type: ignore[union-attr]
            return response.model_dump(mode="json")


# ───────────────────────────────────────────────────────────────
# nimble_web_search_agent_run_start
# ───────────────────────────────────────────────────────────────


class NimbleAgentRunStartToolInput(BaseModel):
    """Input schema for NimbleAgentRunStartTool.

    Starts a resumable Web Search Agent run without waiting for completion.
    """

    agent_id: str = Field(
        description="""The Nimble Web Search Agent id to run.

        Typically a wsa_… id from nimble_web_search_agents_list or
        nimble_web_search_agent_create.
        """,
    )
    input: str = Field(
        description="The research prompt / input for the agent run.",
    )
    effort: AgentEffort | None = Field(
        default=None,
        description="Optional effort level: low, medium, high, x-high, or max.",
    )


class NimbleAgentRunStartTool(_NimbleClientMixin, BaseTool):
    """Start a Nimble Web Search Agent run (does not wait for completion).

    Returns immediately with a run id. Use
    ``nimble_web_search_agent_run_status`` and
    ``nimble_web_search_agent_run_result`` across turns — do not poll inside
    one call.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_web_search_agent_run_start"
    description: str = (
        "Start a Nimble Web Search Agent run (Agent API V2). Returns "
        "immediately with a payload where id is the run_id and "
        "web_search_agent_id is the agent_id — does NOT wait for completion. "
        "Later call nimble_web_search_agent_run_status and "
        "nimble_web_search_agent_run_result across turns. Distinct from "
        "Extract Templates (nimble_extract_template_run)."
    )
    args_schema: type[BaseModel] = NimbleAgentRunStartToolInput
    handle_tool_error: bool = True

    def _build_start_kwargs(
        self,
        agent_id: str,
        input: str,  # noqa: A002
        *,
        effort: AgentEffort | None,
    ) -> dict[str, Any]:
        """Build keyword arguments for agents.runs.create().

        Args:
            agent_id: Web Search Agent id to run.
            input: Research prompt / input.
            effort: Optional effort level.

        Returns:
            Keyword arguments accepted by ``agents.runs.create``.
        """
        kwargs: dict[str, Any] = {
            "agent_id": agent_id,
            "input": input,
        }
        if effort is not None:
            kwargs["effort"] = effort
        return kwargs

    def _run(
        self,
        agent_id: str,
        input: str,  # noqa: A002
        *,
        effort: AgentEffort | None = None,
    ) -> dict[str, Any]:
        """Start an agent run synchronously.

        Args:
            agent_id: Web Search Agent id to run.
            input: Research prompt / input.
            effort: Optional effort level.

        Returns:
            Run create payload (``id`` = run_id, ``web_search_agent_id``).
        """
        require_initialized_client(self.name, self._sync_client, sync=True)

        start_kwargs = self._build_start_kwargs(
            agent_id=agent_id,
            input=input,
            effort=effort,
        )

        with handle_api_errors(operation="agent run start"):
            response = self._sync_client.agents.runs.create(**start_kwargs)  # type: ignore[union-attr]
            return response.model_dump(mode="json")

    async def _arun(
        self,
        agent_id: str,
        input: str,  # noqa: A002
        *,
        effort: AgentEffort | None = None,
    ) -> dict[str, Any]:
        """Start an agent run asynchronously.

        Args:
            agent_id: Web Search Agent id to run.
            input: Research prompt / input.
            effort: Optional effort level.

        Returns:
            Run create payload (``id`` = run_id, ``web_search_agent_id``).
        """
        require_initialized_client(self.name, self._async_client, sync=False)

        start_kwargs = self._build_start_kwargs(
            agent_id=agent_id,
            input=input,
            effort=effort,
        )

        with handle_api_errors(operation="agent run start"):
            response = await self._async_client.agents.runs.create(**start_kwargs)  # type: ignore[union-attr]
            return response.model_dump(mode="json")


# ───────────────────────────────────────────────────────────────
# nimble_web_search_agent_run_status
# ───────────────────────────────────────────────────────────────


class NimbleAgentRunStatusToolInput(BaseModel):
    """Input schema for NimbleAgentRunStatusTool.

    Uses ids from the start response: ``agent_id`` =
    ``web_search_agent_id``, ``run_id`` = ``id``.
    """

    agent_id: str = Field(
        description=(
            "The Nimble Web Search Agent id (often wsa_…). "
            "From start response field web_search_agent_id."
        ),
    )
    run_id: str = Field(
        description=(
            "The run id from nimble_web_search_agent_run_start. "
            "From start response field id."
        ),
    )


class NimbleAgentRunStatusTool(_NimbleClientMixin, BaseTool):
    """Get the status of a Nimble Web Search Agent run.

    Non-terminal statuses (queued/running) are normal resumable results —
    call again later. Use ``nimble_web_search_agent_run_result`` when the run
    is complete.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_web_search_agent_run_status"
    description: str = (
        "Get the status of a Nimble Web Search Agent run (Agent API V2). "
        "Pass agent_id (= start response web_search_agent_id) and run_id "
        "(= start response id). Queued/running statuses are normal — check "
        "again later, then use nimble_web_search_agent_run_result when "
        "complete."
    )
    args_schema: type[BaseModel] = NimbleAgentRunStatusToolInput
    handle_tool_error: bool = True

    def _run(self, agent_id: str, run_id: str) -> dict[str, Any]:
        """Get run status synchronously.

        Args:
            agent_id: Web Search Agent id (``web_search_agent_id``).
            run_id: Run id (start response ``id``).

        Returns:
            Run status payload as a dictionary.
        """
        require_initialized_client(self.name, self._sync_client, sync=True)

        with handle_api_errors(operation="agent run status"):
            response = self._sync_client.agents.runs.get(  # type: ignore[union-attr]
                run_id,
                agent_id=agent_id,
            )
            return response.model_dump(mode="json")

    async def _arun(self, agent_id: str, run_id: str) -> dict[str, Any]:
        """Get run status asynchronously.

        Args:
            agent_id: Web Search Agent id (``web_search_agent_id``).
            run_id: Run id (start response ``id``).

        Returns:
            Run status payload as a dictionary.
        """
        require_initialized_client(self.name, self._async_client, sync=False)

        with handle_api_errors(operation="agent run status"):
            response = await self._async_client.agents.runs.get(  # type: ignore[union-attr]
                run_id,
                agent_id=agent_id,
            )
            return response.model_dump(mode="json")


# ───────────────────────────────────────────────────────────────
# nimble_web_search_agent_run_result
# ───────────────────────────────────────────────────────────────


class NimbleAgentRunResultToolInput(BaseModel):
    """Input schema for NimbleAgentRunResultTool.

    Uses ids from the start response: ``agent_id`` =
    ``web_search_agent_id``, ``run_id`` = ``id``.
    """

    agent_id: str = Field(
        description=(
            "The Nimble Web Search Agent id (often wsa_…). "
            "From start response field web_search_agent_id."
        ),
    )
    run_id: str = Field(
        description=(
            "The run id from nimble_web_search_agent_run_start. "
            "From start response field id."
        ),
    )


class NimbleAgentRunResultTool(_NimbleClientMixin, BaseTool):
    """Get the result of a completed Nimble Web Search Agent run.

    Prefer calling after ``nimble_web_search_agent_run_status`` indicates
    completion. Surfaces structured results, citations/trust metadata when
    returned. Failed runs raise ``ToolException``.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_web_search_agent_run_result"
    description: str = (
        "Get the result of a completed Nimble Web Search Agent run "
        "(Agent API V2). Pass agent_id (= start response "
        "web_search_agent_id) and run_id (= start response id). Use after "
        "status shows the run finished. May include text/JSON, citations, "
        "and trust metadata when available."
    )
    args_schema: type[BaseModel] = NimbleAgentRunResultToolInput
    handle_tool_error: bool = True

    def _normalize_result(self, response: Any) -> dict[str, Any]:
        """Normalize result payloads or raise on failed runs.

        Args:
            response: SDK run result union payload
                (``TaskRunResultPublicV2`` or ``TaskRunFailedResultPublicV2``).

        Returns:
            Successful result as a JSON-serializable dictionary.

        Raises:
            ToolException: If the run failed (``error`` present, no ``output``).
        """
        payload = response.model_dump(mode="json")
        if (
            isinstance(payload, dict)
            and payload.get("error") is not None
            and "output" not in payload
        ):
            msg = f"Agent run failed: {payload['error']}"
            raise ToolException(msg)
        return payload

    def _run(self, agent_id: str, run_id: str) -> dict[str, Any]:
        """Get run result synchronously.

        Args:
            agent_id: Web Search Agent id (``web_search_agent_id``).
            run_id: Run id (start response ``id``).

        Returns:
            Successful run result payload as a dictionary.
        """
        require_initialized_client(self.name, self._sync_client, sync=True)

        with handle_api_errors(operation="agent run result"):
            response = self._sync_client.agents.runs.result(  # type: ignore[union-attr]
                run_id,
                agent_id=agent_id,
            )
            return self._normalize_result(response)

    async def _arun(self, agent_id: str, run_id: str) -> dict[str, Any]:
        """Get run result asynchronously.

        Args:
            agent_id: Web Search Agent id (``web_search_agent_id``).
            run_id: Run id (start response ``id``).

        Returns:
            Successful run result payload as a dictionary.
        """
        require_initialized_client(self.name, self._async_client, sync=False)

        with handle_api_errors(operation="agent run result"):
            response = await self._async_client.agents.runs.result(  # type: ignore[union-attr]
                run_id,
                agent_id=agent_id,
            )
            return self._normalize_result(response)
