"""NimbleToolkit providing all Nimble tools for LangChain agents."""

from __future__ import annotations

import warnings

from langchain_core.tools import BaseTool, BaseToolkit
from langchain_core.utils import secret_from_env
from pydantic import Field, SecretStr


class NimbleToolkit(BaseToolkit):
    """Toolkit providing all Nimble API tools for LangChain agents.

    Use ``include_*`` flags to control which tools are returned.
    All tools share the same API key and client configuration.

    By default, only Search and Extract tools are enabled. Crawl, Map,
    Extract Templates, and Agent API V2 tools are opt-in.

    Example::

        toolkit = NimbleToolkit(api_key="your-key", include_map=True)
        tools = toolkit.get_tools()
        # Returns [NimbleSearchTool, NimbleExtractTool, NimbleMapTool]
    """

    nimble_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env("NIMBLE_API_KEY", default=""),
        description="API key for Nimbleway (or set NIMBLE_API_KEY env var).",
    )
    nimble_api_url: str | None = Field(
        alias="base_url",
        default=None,
        description="Override base URL for the Nimble API.",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retry attempts for 5xx errors.",
    )

    include_search: bool = Field(
        default=True,
        description="Include NimbleSearchTool.",
    )
    include_extract: bool = Field(
        default=True,
        description="Include NimbleExtractTool.",
    )
    include_crawl: bool = Field(
        default=False,
        description="Include NimbleCrawlTool.",
    )
    include_map: bool = Field(
        default=False,
        description="Include NimbleMapTool.",
    )
    include_extract_templates: bool = Field(
        default=False,
        description="Include Extract Templates tools (list, get, run).",
    )
    include_web_search_agents: bool = Field(
        default=False,
        description=(
            "Include Agent API V2 Web Search Agent tools "
            "(list, create, start/status/result)."
        ),
    )
    include_agent: bool = Field(
        default=False,
        description=(
            "Deprecated. Include legacy nimble_agent_* aliases that wrap "
            "Extract Templates. Prefer include_extract_templates."
        ),
    )

    crawl_polling_interval: float = Field(
        default=5.0,
        gt=0,
        description="Seconds between crawl status polls.",
    )
    crawl_timeout: float = Field(
        default=300.0,
        gt=0,
        description="Maximum seconds to wait for crawl completion.",
    )

    def get_tools(self) -> list[BaseTool]:
        """Get the selected Nimble tools.

        Returns:
            List of BaseTool instances based on the enabled ``include_*`` flags.
        """
        common_kwargs: dict[str, object] = {
            "api_key": self.nimble_api_key.get_secret_value(),
            "max_retries": self.max_retries,
        }
        if self.nimble_api_url is not None:
            common_kwargs["base_url"] = self.nimble_api_url

        tools: list[BaseTool] = []

        if self.include_search:
            from langchain_nimble.tools.search_tool import NimbleSearchTool

            tools.append(NimbleSearchTool(**common_kwargs))

        if self.include_extract:
            from langchain_nimble.tools.extract_tool import NimbleExtractTool

            tools.append(NimbleExtractTool(**common_kwargs))

        if self.include_crawl:
            from langchain_nimble.tools.crawl_tool import NimbleCrawlTool

            tools.append(
                NimbleCrawlTool(
                    **common_kwargs,
                    polling_interval=self.crawl_polling_interval,
                    timeout=self.crawl_timeout,
                )
            )

        if self.include_map:
            from langchain_nimble.tools.map_tool import NimbleMapTool

            tools.append(NimbleMapTool(**common_kwargs))

        if self.include_agent:
            warnings.warn(
                "NimbleToolkit(include_agent=True) is deprecated. "
                "Use include_extract_templates=True for Extract Templates, "
                "or include_web_search_agents=True for Agent API V2 research "
                "tools.",
                DeprecationWarning,
                stacklevel=2,
            )

        if self.include_extract_templates:
            from langchain_nimble.tools.extract_template_tool import (
                NimbleExtractTemplateGetTool,
                NimbleExtractTemplateListTool,
                NimbleExtractTemplateRunTool,
            )

            tools.extend(
                [
                    NimbleExtractTemplateListTool(**common_kwargs),
                    NimbleExtractTemplateGetTool(**common_kwargs),
                    NimbleExtractTemplateRunTool(**common_kwargs),
                ]
            )
        elif self.include_agent:
            from langchain_nimble.tools.agent_tool import (
                NimbleAgentGetTool,
                NimbleAgentListTool,
                NimbleAgentRunTool,
            )

            tools.extend(
                [
                    NimbleAgentListTool(**common_kwargs),
                    NimbleAgentGetTool(**common_kwargs),
                    NimbleAgentRunTool(**common_kwargs),
                ]
            )

        if self.include_web_search_agents:
            from langchain_nimble.tools.agents_v2_tool import (
                NimbleAgentCreateTool,
                NimbleAgentRunResultTool,
                NimbleAgentRunStartTool,
                NimbleAgentRunStatusTool,
                NimbleAgentsListTool,
                NimbleAgentTemplatesListTool,
            )

            tools.extend(
                [
                    NimbleAgentsListTool(**common_kwargs),
                    NimbleAgentTemplatesListTool(**common_kwargs),
                    NimbleAgentCreateTool(**common_kwargs),
                    NimbleAgentRunStartTool(**common_kwargs),
                    NimbleAgentRunStatusTool(**common_kwargs),
                    NimbleAgentRunResultTool(**common_kwargs),
                ]
            )

        return tools
