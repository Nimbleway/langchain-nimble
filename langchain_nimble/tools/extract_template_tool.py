"""LangChain tools for Nimble Extract Templates API."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_nimble._utilities import _NimbleClientMixin, handle_api_errors

# ───────────────────────────────────────────────────────────────
# nimble_extract_template_list
# ───────────────────────────────────────────────────────────────


class NimbleExtractTemplateListToolInput(BaseModel):
    """Input schema for NimbleExtractTemplateListTool."""

    limit: int | None = Field(
        default=None,
        description="Maximum number of extract templates to return.",
    )
    offset: int | None = Field(
        default=None,
        description="Pagination offset for extract template listing.",
    )


class NimbleExtractTemplateListTool(_NimbleClientMixin, BaseTool):
    """List available Nimble Extract Templates.

    Returns template names and metadata. Use this tool first to discover
    which templates are available before getting details or running one.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_extract_template_list"
    description: str = (
        "List available Nimble Extract Templates for structured site scraping. "
        "Returns template names and metadata. Use this first to discover which "
        "templates exist, then use nimble_extract_template_get to see required "
        "parameters. This is NOT Agent API V2 research — use nimble_agents_list "
        "for Web Search Agents."
    )
    args_schema: type[BaseModel] = NimbleExtractTemplateListToolInput
    handle_tool_error: bool = True

    def _build_list_kwargs(
        self,
        *,
        limit: int | None,
        offset: int | None,
    ) -> dict[str, Any]:
        """Build keyword arguments for extract.templates.list()."""
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
        """List extract templates synchronously."""
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        list_kwargs = self._build_list_kwargs(limit=limit, offset=offset)

        with handle_api_errors(operation="extract template list"):
            response = self._sync_client.extract.templates.list(**list_kwargs)
            return [item.model_dump(mode="json") for item in response.items]

    async def _arun(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        """List extract templates asynchronously."""
        if self._async_client is None:
            msg = "Async client not initialized"
            raise RuntimeError(msg)

        list_kwargs = self._build_list_kwargs(limit=limit, offset=offset)

        with handle_api_errors(operation="extract template list"):
            response = await self._async_client.extract.templates.list(**list_kwargs)
            return [item.model_dump(mode="json") for item in response.items]


# ───────────────────────────────────────────────────────────────
# nimble_extract_template_get
# ───────────────────────────────────────────────────────────────


class NimbleExtractTemplateGetToolInput(BaseModel):
    """Input schema for NimbleExtractTemplateGetTool."""

    template_name: str = Field(
        description="""The extract template name to get details for.

        Use nimble_extract_template_list first to discover available template
        names, then pass one here to see its schema and versions.
        Examples: "amazon_pdp", "google_search", "walmart_pdp"
        """,
    )


class NimbleExtractTemplateGetTool(_NimbleClientMixin, BaseTool):
    """Get details about a specific Nimble Extract Template.

    Returns template metadata including published version information.
    Use after nimble_extract_template_list before running a template.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_extract_template_get"
    description: str = (
        "Get details about a Nimble Extract Template including its published "
        "version and metadata. Use after nimble_extract_template_list to learn "
        "what params to pass to nimble_extract_template_run. This is structured "
        "site scraping, not Agent API V2 research."
    )
    args_schema: type[BaseModel] = NimbleExtractTemplateGetToolInput
    handle_tool_error: bool = True

    def _run(self, template_name: str) -> dict[str, Any]:
        """Get extract template details synchronously."""
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        with handle_api_errors(operation="extract template get"):
            response = self._sync_client.extract.templates.get(template_name)
            return response.model_dump(mode="json")

    async def _arun(self, template_name: str) -> dict[str, Any]:
        """Get extract template details asynchronously."""
        if self._async_client is None:
            msg = "Async client not initialized"
            raise RuntimeError(msg)

        with handle_api_errors(operation="extract template get"):
            response = await self._async_client.extract.templates.get(template_name)
            return response.model_dump(mode="json")


# ───────────────────────────────────────────────────────────────
# nimble_extract_template_run
# ───────────────────────────────────────────────────────────────


class NimbleExtractTemplateRunToolInput(BaseModel):
    """Input schema for NimbleExtractTemplateRunTool."""

    template: str = Field(
        description="""The extract template name to run.

        Use nimble_extract_template_list to discover available templates, then
        nimble_extract_template_get to inspect metadata before running.
        Examples: "amazon_pdp", "google_search", "walmart_pdp"
        """,
    )
    params: dict[str, object] = Field(
        description="""Template-specific parameters.

        Each template requires different parameters. Inspect the template
        (or your Nimble console / docs) for the exact params.

        Common examples:
        - amazon_pdp: {"asin": "B0..."}
        - google_search: {"query": "search term"}
        - walmart_pdp: {"url": "https://walmart.com/ip/..."}
        """,
    )
    localization: bool | None = Field(
        default=None,
        description="Enable localization for template results.",
    )


class NimbleExtractTemplateRunTool(_NimbleClientMixin, BaseTool):
    """Run a Nimble Extract Template for structured data collection.

    Extract Templates handle structured site scraping workflows such as
    product page parsing and search result extraction. This is distinct from
    Agent API V2 research agents (use nimble_agent_run_start / status / result).

    Recommended workflow:
    1. nimble_extract_template_list → discover available templates
    2. nimble_extract_template_get → inspect a specific template
    3. nimble_extract_template_run → execute with the correct params

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_extract_template_run"
    description: str = (
        "Run a Nimble Extract Template for structured site scraping. "
        "Use nimble_extract_template_list and nimble_extract_template_get "
        "first to discover templates and their parameters. Do not use this "
        "for multi-minute research — use Agent API V2 run tools instead."
    )
    args_schema: type[BaseModel] = NimbleExtractTemplateRunToolInput
    handle_tool_error: bool = True

    def _build_run_kwargs(
        self,
        template: str,
        params: dict[str, object],
        *,
        localization: bool | None,
    ) -> dict[str, Any]:
        """Build keyword arguments for extract.templates.run()."""
        kwargs: dict[str, Any] = {
            "template": template,
            "params": params,
        }
        if localization is not None:
            kwargs["localization"] = localization
        return kwargs

    def _validate_response(self, response: Any) -> dict[str, Any]:
        """Validate template response status and return dumped data."""
        if response.status != "success":
            warnings_msg = ""
            if response.warnings:
                warnings_msg = f" Warnings: {response.warnings}"
            msg = (
                f"Extract template returned status '{response.status}' "
                f"for task {response.task_id}.{warnings_msg}"
            )
            raise ToolException(msg)
        return response.model_dump(mode="json")

    def _run(
        self,
        template: str,
        params: dict[str, object],
        *,
        localization: bool | None = None,
    ) -> dict[str, Any]:
        """Execute extract template synchronously."""
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        run_kwargs = self._build_run_kwargs(
            template=template,
            params=params,
            localization=localization,
        )

        with handle_api_errors(operation="extract template run"):
            response = self._sync_client.extract.templates.run(**run_kwargs)
            return self._validate_response(response)

    async def _arun(
        self,
        template: str,
        params: dict[str, object],
        *,
        localization: bool | None = None,
    ) -> dict[str, Any]:
        """Execute extract template asynchronously."""
        if self._async_client is None:
            msg = "Async client not initialized"
            raise RuntimeError(msg)

        run_kwargs = self._build_run_kwargs(
            template=template,
            params=params,
            localization=localization,
        )

        with handle_api_errors(operation="extract template run"):
            response = await self._async_client.extract.templates.run(**run_kwargs)
            return self._validate_response(response)
