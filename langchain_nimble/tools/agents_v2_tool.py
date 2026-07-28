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
    configuration (Mode 2 bootstrap).
    """

    template: str | None = Field(
        default=None,
        description="Optional agent template name to create from.",
    )
    agent_name: str | None = Field(
        default=None,
        description="Optional unique agent name (409 if already taken on create).",
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
        description="""Primary use case — locked at create time.

        - research: free-form cited answer (output.type=text)
        - enrichment: fill input_data against a schema (output.type=json)
        - dataset_building: structured table from scratch (output.type=json)

        Cannot be changed later via run overrides (mismatch → 422).
        """,
    )
    skill: str | None = Field(
        default=None,
        description=(
            "Operating instructions / domain expertise for the agent "
            "(also known as domain_expertise)."
        ),
    )
    sources: dict[str, Any] | None = Field(
        default=None,
        description="""Source guidance persisted on the agent.

        Fields: allow/block (groups with title, domains, order) and
        avoid/prioritize (free-text strings).
        """,
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description="JSON Schema for structured output (enrichment/dataset_building).",
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
        skill: str | None,
        sources: dict[str, Any] | None,
        output_schema: dict[str, Any] | None,
        goals: list[str] | None,
    ) -> dict[str, Any]:
        """Build keyword arguments for agents.create().

        Args:
            template: Optional template name to create from.
            agent_name: Optional unique agent name.
            display_name: Optional display name.
            description: Optional description.
            effort: Optional default effort level.
            use_case: Optional use case (locked after create).
            skill: Optional domain expertise / operating instructions.
            sources: Optional source guidance to persist.
            output_schema: Optional JSON Schema to persist.
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
        if skill is not None:
            kwargs["skill"] = skill
        if sources is not None:
            kwargs["sources"] = sources
        if output_schema is not None:
            kwargs["output_schema"] = output_schema
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
        skill: str | None = None,
        sources: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        goals: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create an agent synchronously.

        Args:
            template: Optional template name to create from.
            agent_name: Optional unique agent name.
            display_name: Optional display name.
            description: Optional description.
            effort: Optional default effort level.
            use_case: Optional use case (locked after create).
            skill: Optional domain expertise / operating instructions.
            sources: Optional source guidance to persist.
            output_schema: Optional JSON Schema to persist.
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
            skill=skill,
            sources=sources,
            output_schema=output_schema,
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
        skill: str | None = None,
        sources: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        goals: list[str] | None = None,
    ) -> dict[str, Any]:
        """Create an agent asynchronously.

        Args:
            template: Optional template name to create from.
            agent_name: Optional unique agent name.
            display_name: Optional display name.
            description: Optional description.
            effort: Optional default effort level.
            use_case: Optional use case (locked after create).
            skill: Optional domain expertise / operating instructions.
            sources: Optional source guidance to persist.
            output_schema: Optional JSON Schema to persist.
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
            skill=skill,
            sources=sources,
            output_schema=output_schema,
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
    Supports Mode 1 (``agent_name``), Mode 2 (``agent_id``), and Mode 3
    (neither — anonymous one-shot).
    """

    input: str = Field(
        description="The research prompt / input for the agent run.",
    )
    agent_id: str | None = Field(
        default=None,
        description="""Mode 2: existing Web Search Agent id (wsa_…).

        From nimble_web_search_agents_list / create, or from a prior start
        response field web_search_agent_id. Prefer this when the host can
        persist the id.
        """,
    )
    agent_name: str | None = Field(
        default=None,
        description="""Mode 1 (default for stateless hosts): create-or-reuse by name.

        First unseen name creates the agent; later calls reuse it. Response
        always includes web_search_agent_id. Reusing a name does not brick
        after a failed first run. Ignored when agent_id is set.
        """,
    )
    use_case: AgentUseCase | None = Field(
        default=None,
        description="""Use case — locked on the agent after create.

        research → text; enrichment → json (needs input_data); dataset_building
        → json. On an existing agent omit or pass the same value; a different
        value returns 422. On Mode 1 first create / Mode 3 it is stored.
        """,
    )
    skill: str | None = Field(
        default=None,
        description="""Domain expertise / operating instructions for this run.

        One-time override on an existing agent; persisted on Mode 1 first
        create. Alias of domain_expertise.
        """,
    )
    effort: AgentEffort | None = Field(
        default=None,
        description="""Effort tier for this run: low | medium | high | x-high | max.

        Runs often take 3-15 minutes (medium ~90-160s observed; low ~15-17s
        may skip live research). Prefer medium+ for real research. Do not
        poll inside one tool call — use status/result across turns.
        """,
    )
    sources: dict[str, Any] | None = Field(
        default=None,
        description="""Source guidance override for this run.

        allow/block: groups [{title, domains, order}]; avoid/prioritize:
        free-text strings. Persisted only on Mode 1 first create; otherwise
        one-time.
        """,
    )
    output_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "JSON Schema override for structured output. Persisted only on "
            "Mode 1 first create; otherwise one-time."
        ),
    )
    input_data: list[dict[str, Any]] | dict[str, Any] | None = Field(
        default=None,
        description="""Enrichment payload only (never stored on the agent).

        Partial rows or a single object mirroring output_schema. Distinct
        from output_schema (data vs shape).
        """,
    )
    enable_events: bool | None = Field(
        default=None,
        description=(
            "If true, enable SSE progress on GET .../runs/{run_id}/events. "
            "This package does not expose an events tool yet."
        ),
    )
    previous_interaction_id: str | None = Field(
        default=None,
        description="Optional prior interaction id to continue a conversation.",
    )


class NimbleAgentRunStartTool(_NimbleClientMixin, BaseTool):
    """Start a Nimble Web Search Agent run (does not wait for completion).

    LangChain agents are typically **stateless** across turns unless the app
    persists ids, so prefer **Mode 1** (``agent_name`` create-or-reuse). Use
    **Mode 2** (``agent_id``) when ``wsa_…`` is persisted. Omit both for
    **Mode 3** anonymous one-shot (still returns ``web_search_agent_id``).

    Returns immediately with ``id`` (``task_run_…``) and
    ``web_search_agent_id`` (``wsa_…``). Use status/result across turns —
    do not poll inside one call. Runs may take 3-15 minutes.

    ``nimble_python`` 1.1.x types Mode 2 as ``agents.runs.create`` and Mode
    1/3 as ``agents.run``; ``agent_name`` / ``use_case`` / ``skill`` on the
    run body are sent via ``extra_body`` until the SDK exposes them.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_web_search_agent_run_start"
    description: str = (
        "Start a Nimble Web Search Agent run (Agent API V2). Returns "
        "immediately — does NOT wait for completion (often 3-15 minutes). "
        "Modes: (1) agent_name create-or-reuse for stateless hosts; "
        "(2) agent_id when you persist wsa_…; (3) omit both for anonymous. "
        "Response: id=run_id (task_run_…), web_search_agent_id=agent_id. "
        "Then use nimble_web_search_agent_run_status / _run_result across "
        "turns. Expose skill, use_case, effort, sources, output_schema, "
        "input_data as needed. Distinct from Extract Templates."
    )
    args_schema: type[BaseModel] = NimbleAgentRunStartToolInput
    handle_tool_error: bool = True

    def _build_typed_run_kwargs(
        self,
        input: str,  # noqa: A002
        *,
        effort: AgentEffort | None,
        sources: dict[str, Any] | None,
        output_schema: dict[str, Any] | None,
        input_data: list[dict[str, Any]] | dict[str, Any] | None,
        enable_events: bool | None,
        previous_interaction_id: str | None,
    ) -> dict[str, Any]:
        """Build kwargs already typed on the SDK run helpers.

        Args:
            input: Research prompt / input.
            effort: Optional effort level.
            sources: Optional source guidance.
            output_schema: Optional JSON Schema override.
            input_data: Optional enrichment payload.
            enable_events: Optional SSE flag.
            previous_interaction_id: Optional conversation continuation id.

        Returns:
            Keyword arguments shared by ``agents.run`` and ``agents.runs.create``.
        """
        kwargs: dict[str, Any] = {"input": input}
        if effort is not None:
            kwargs["effort"] = effort
        if sources is not None:
            kwargs["sources"] = sources
        if output_schema is not None:
            kwargs["output_schema"] = output_schema
        if input_data is not None:
            kwargs["input_data"] = input_data
        if enable_events is not None:
            kwargs["enable_events"] = enable_events
        if previous_interaction_id is not None:
            kwargs["previous_interaction_id"] = previous_interaction_id
        return kwargs

    def _build_extra_body(
        self,
        *,
        agent_id: str | None,
        agent_name: str | None,
        use_case: AgentUseCase | None,
        skill: str | None,
    ) -> dict[str, Any] | None:
        """Build extra_body for fields not yet typed on SDK run methods.

        Args:
            agent_id: Mode 2 agent id (when set, agent_name is ignored).
            agent_name: Mode 1 create-or-reuse name.
            use_case: Optional use case (locked after create).
            skill: Optional domain expertise override.

        Returns:
            Extra JSON body fields, or ``None`` when empty.
        """
        extra: dict[str, Any] = {}
        if agent_id is None and agent_name is not None:
            extra["agent_name"] = agent_name
        if use_case is not None:
            extra["use_case"] = use_case
        if skill is not None:
            extra["skill"] = skill
        return extra or None

    def _run(
        self,
        input: str,  # noqa: A002
        *,
        agent_id: str | None = None,
        agent_name: str | None = None,
        use_case: AgentUseCase | None = None,
        skill: str | None = None,
        effort: AgentEffort | None = None,
        sources: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        input_data: list[dict[str, Any]] | dict[str, Any] | None = None,
        enable_events: bool | None = None,
        previous_interaction_id: str | None = None,
    ) -> dict[str, Any]:
        """Start an agent run synchronously (Mode 1 / 2 / 3).

        Args:
            input: Research prompt / input.
            agent_id: Mode 2 agent id.
            agent_name: Mode 1 create-or-reuse name.
            use_case: Optional use case.
            skill: Optional domain expertise.
            effort: Optional effort level.
            sources: Optional source guidance.
            output_schema: Optional JSON Schema override.
            input_data: Optional enrichment payload.
            enable_events: Optional SSE flag.
            previous_interaction_id: Optional conversation continuation id.

        Returns:
            Run create payload (``id`` = run_id, ``web_search_agent_id``).
        """
        require_initialized_client(self.name, self._sync_client, sync=True)

        typed_kwargs = self._build_typed_run_kwargs(
            input=input,
            effort=effort,
            sources=sources,
            output_schema=output_schema,
            input_data=input_data,
            enable_events=enable_events,
            previous_interaction_id=previous_interaction_id,
        )
        extra_body = self._build_extra_body(
            agent_id=agent_id,
            agent_name=agent_name,
            use_case=use_case,
            skill=skill,
        )
        if extra_body is not None:
            typed_kwargs["extra_body"] = extra_body

        with handle_api_errors(operation="agent run start"):
            if agent_id:
                response = self._sync_client.agents.runs.create(  # type: ignore[union-attr]
                    agent_id,
                    **typed_kwargs,
                )
            else:
                response = self._sync_client.agents.run(**typed_kwargs)  # type: ignore[union-attr]
            return response.model_dump(mode="json")

    async def _arun(
        self,
        input: str,  # noqa: A002
        *,
        agent_id: str | None = None,
        agent_name: str | None = None,
        use_case: AgentUseCase | None = None,
        skill: str | None = None,
        effort: AgentEffort | None = None,
        sources: dict[str, Any] | None = None,
        output_schema: dict[str, Any] | None = None,
        input_data: list[dict[str, Any]] | dict[str, Any] | None = None,
        enable_events: bool | None = None,
        previous_interaction_id: str | None = None,
    ) -> dict[str, Any]:
        """Start an agent run asynchronously (Mode 1 / 2 / 3).

        Args:
            input: Research prompt / input.
            agent_id: Mode 2 agent id.
            agent_name: Mode 1 create-or-reuse name.
            use_case: Optional use case.
            skill: Optional domain expertise.
            effort: Optional effort level.
            sources: Optional source guidance.
            output_schema: Optional JSON Schema override.
            input_data: Optional enrichment payload.
            enable_events: Optional SSE flag.
            previous_interaction_id: Optional conversation continuation id.

        Returns:
            Run create payload (``id`` = run_id, ``web_search_agent_id``).
        """
        require_initialized_client(self.name, self._async_client, sync=False)

        typed_kwargs = self._build_typed_run_kwargs(
            input=input,
            effort=effort,
            sources=sources,
            output_schema=output_schema,
            input_data=input_data,
            enable_events=enable_events,
            previous_interaction_id=previous_interaction_id,
        )
        extra_body = self._build_extra_body(
            agent_id=agent_id,
            agent_name=agent_name,
            use_case=use_case,
            skill=skill,
        )
        if extra_body is not None:
            typed_kwargs["extra_body"] = extra_body

        with handle_api_errors(operation="agent run start"):
            if agent_id:
                response = await self._async_client.agents.runs.create(  # type: ignore[union-attr]
                    agent_id,
                    **typed_kwargs,
                )
            else:
                response = await self._async_client.agents.run(**typed_kwargs)  # type: ignore[union-attr]
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
