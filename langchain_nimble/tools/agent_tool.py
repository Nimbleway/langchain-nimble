"""Deprecated aliases for legacy Nimble Agent tools.

These tools previously wrapped the deprecated site-scraping ``client.agent.*``
API. They now delegate to Extract Templates (``client.extract.templates.*``)
and emit ``DeprecationWarning``. Prefer the ``nimble_extract_template_*`` tools.

They are intentionally NOT wired to Agent API V2 research agents.
"""

from __future__ import annotations

import warnings
from typing import Any

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_nimble._utilities import (
    _NimbleClientMixin,
    handle_api_errors,
    require_initialized_client,
)

_DEPRECATION_MESSAGE = (
    "NimbleAgent* tools are deprecated and now wrap Extract Templates. "
    "Use nimble_extract_template_list / nimble_extract_template_get / "
    "nimble_extract_template_run instead. For Agent API V2 research agents, "
    "use nimble_web_search_agents_list, nimble_web_search_agent_run_start, "
    "nimble_web_search_agent_run_status, and nimble_web_search_agent_run_result."
)


def _warn_deprecated() -> None:
    """Emit a deprecation warning for legacy agent tool names."""
    warnings.warn(_DEPRECATION_MESSAGE, DeprecationWarning, stacklevel=3)


# ───────────────────────────────────────────────────────────────
# nimble_agent_list (deprecated → extract templates list)
# ───────────────────────────────────────────────────────────────


class NimbleAgentListToolInput(BaseModel):
    """Input schema for NimbleAgentListTool.

    Deprecated alias of Extract Templates list pagination fields.
    """

    limit: int | None = Field(
        default=None,
        description="Maximum number of extract templates to return.",
    )
    offset: int | None = Field(
        default=None,
        description="Pagination offset for extract template listing.",
    )


class NimbleAgentListTool(_NimbleClientMixin, BaseTool):
    """Deprecated: list Extract Templates under the legacy agent tool name.

    Prefer ``NimbleExtractTemplateListTool``.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_agent_list"
    description: str = (
        "DEPRECATED: List Nimble Extract Templates (legacy agent tool name). "
        "Prefer nimble_extract_template_list. For Agent API V2 research, use "
        "nimble_web_search_agents_list instead."
    )
    args_schema: type[BaseModel] = NimbleAgentListToolInput
    handle_tool_error: bool = True

    def _build_list_kwargs(
        self,
        *,
        limit: int | None,
        offset: int | None,
    ) -> dict[str, Any]:
        """Build keyword arguments for extract.templates.list().

        Args:
            limit: Maximum number of templates to return.
            offset: Pagination offset.

        Returns:
            Keyword arguments accepted by ``extract.templates.list``.
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
        """List extract templates synchronously (deprecated alias).

        Args:
            limit: Maximum number of templates to return.
            offset: Pagination offset.

        Returns:
            List of template records as dictionaries.
        """
        _warn_deprecated()
        require_initialized_client(self.name, self._sync_client, sync=True)

        list_kwargs = self._build_list_kwargs(limit=limit, offset=offset)

        with handle_api_errors(operation="extract template list"):
            response = self._sync_client.extract.templates.list(**list_kwargs)  # type: ignore[union-attr]
            return [item.model_dump(mode="json") for item in response.items]

    async def _arun(
        self,
        *,
        limit: int | None = None,
        offset: int | None = None,
    ) -> list[dict[str, Any]]:
        """List extract templates asynchronously (deprecated alias).

        Args:
            limit: Maximum number of templates to return.
            offset: Pagination offset.

        Returns:
            List of template records as dictionaries.
        """
        _warn_deprecated()
        require_initialized_client(self.name, self._async_client, sync=False)

        list_kwargs = self._build_list_kwargs(limit=limit, offset=offset)

        with handle_api_errors(operation="extract template list"):
            response = await self._async_client.extract.templates.list(**list_kwargs)  # type: ignore[union-attr]
            return [item.model_dump(mode="json") for item in response.items]


# ───────────────────────────────────────────────────────────────
# nimble_agent_get (deprecated → extract templates get)
# ───────────────────────────────────────────────────────────────


class NimbleAgentGetToolInput(BaseModel):
    """Input schema for NimbleAgentGetTool.

    Deprecated alias of Extract Templates get input.
    """

    template_name: str = Field(
        description="""The extract template name to get details for.

        Prefer nimble_extract_template_get. Examples: "amazon_pdp",
        "google_search", "walmart_pdp"
        """,
    )


class NimbleAgentGetTool(_NimbleClientMixin, BaseTool):
    """Deprecated: get Extract Template details under the legacy agent name.

    Prefer ``NimbleExtractTemplateGetTool``.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_agent_get"
    description: str = (
        "DEPRECATED: Get a Nimble Extract Template (legacy agent tool name). "
        "Prefer nimble_extract_template_get."
    )
    args_schema: type[BaseModel] = NimbleAgentGetToolInput
    handle_tool_error: bool = True

    def _run(self, template_name: str) -> dict[str, Any]:
        """Get extract template details synchronously (deprecated alias).

        Args:
            template_name: Extract template name to fetch.

        Returns:
            Template metadata as a dictionary.
        """
        _warn_deprecated()
        require_initialized_client(self.name, self._sync_client, sync=True)

        with handle_api_errors(operation="extract template get"):
            response = self._sync_client.extract.templates.get(template_name)  # type: ignore[union-attr]
            return response.model_dump(mode="json")

    async def _arun(self, template_name: str) -> dict[str, Any]:
        """Get extract template details asynchronously (deprecated alias).

        Args:
            template_name: Extract template name to fetch.

        Returns:
            Template metadata as a dictionary.
        """
        _warn_deprecated()
        require_initialized_client(self.name, self._async_client, sync=False)

        with handle_api_errors(operation="extract template get"):
            response = await self._async_client.extract.templates.get(template_name)  # type: ignore[union-attr]
            return response.model_dump(mode="json")


# ───────────────────────────────────────────────────────────────
# nimble_agent_run (deprecated → extract templates run)
# ───────────────────────────────────────────────────────────────


class NimbleAgentRunToolInput(BaseModel):
    """Input schema for NimbleAgentRunTool.

    Deprecated alias of Extract Templates run input (``agent`` = template).
    """

    agent: str = Field(
        description="""The extract template name to run (legacy param name).

        Prefer nimble_extract_template_run with template=. Examples:
        "amazon_pdp", "google_search", "walmart_pdp"
        """,
    )
    params: dict[str, object] = Field(
        description="""Template-specific parameters.

        Prefer nimble_extract_template_run. Common examples:
        - amazon_pdp: {"asin": "B0..."}
        - google_search: {"query": "search term"}
        """,
    )
    localization: bool | None = Field(
        default=None,
        description="Enable localization for template results.",
    )


class NimbleAgentRunTool(_NimbleClientMixin, BaseTool):
    """Deprecated: run an Extract Template under the legacy agent tool name.

    Prefer ``NimbleExtractTemplateRunTool``.
    This does NOT call Agent API V2 research agents.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
    """

    name: str = "nimble_agent_run"
    description: str = (
        "DEPRECATED: Run a Nimble Extract Template (legacy agent tool name). "
        "Prefer nimble_extract_template_run. For Agent API V2 research, use "
        "nimble_web_search_agent_run_start / status / result."
    )
    args_schema: type[BaseModel] = NimbleAgentRunToolInput
    handle_tool_error: bool = True

    def _build_run_kwargs(
        self,
        agent: str,
        params: dict[str, object],
        *,
        localization: bool | None,
    ) -> dict[str, Any]:
        """Build keyword arguments for extract.templates.run().

        Args:
            agent: Extract template name (legacy param name).
            params: Template-specific parameters.
            localization: Optional localization flag.

        Returns:
            Keyword arguments accepted by ``extract.templates.run``.
        """
        kwargs: dict[str, Any] = {
            "template": agent,
            "params": params,
        }
        if localization is not None:
            kwargs["localization"] = localization
        return kwargs

    def _validate_response(self, response: Any) -> dict[str, Any]:
        """Validate template response status and return dumped data.

        Args:
            response: SDK extract template run response.

        Returns:
            Successful response as a JSON-serializable dictionary.

        Raises:
            ToolException: If response status is not ``success``.
        """
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
        agent: str,
        params: dict[str, object],
        *,
        localization: bool | None = None,
    ) -> dict[str, Any]:
        """Execute extract template synchronously (deprecated alias).

        Args:
            agent: Extract template name (legacy param name).
            params: Template-specific parameters.
            localization: Optional localization flag.

        Returns:
            Successful template run payload as a dictionary.
        """
        _warn_deprecated()
        require_initialized_client(self.name, self._sync_client, sync=True)

        run_kwargs = self._build_run_kwargs(
            agent=agent,
            params=params,
            localization=localization,
        )

        with handle_api_errors(operation="extract template run"):
            response = self._sync_client.extract.templates.run(**run_kwargs)  # type: ignore[union-attr]
            return self._validate_response(response)

    async def _arun(
        self,
        agent: str,
        params: dict[str, object],
        *,
        localization: bool | None = None,
    ) -> dict[str, Any]:
        """Execute extract template asynchronously (deprecated alias).

        Args:
            agent: Extract template name (legacy param name).
            params: Template-specific parameters.
            localization: Optional localization flag.

        Returns:
            Successful template run payload as a dictionary.
        """
        _warn_deprecated()
        require_initialized_client(self.name, self._async_client, sync=False)

        run_kwargs = self._build_run_kwargs(
            agent=agent,
            params=params,
            localization=localization,
        )

        with handle_api_errors(operation="extract template run"):
            response = await self._async_client.extract.templates.run(**run_kwargs)  # type: ignore[union-attr]
            return self._validate_response(response)
