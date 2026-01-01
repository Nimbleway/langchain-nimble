"""LangChain search tool for Nimble Search API."""

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field, model_validator

from ._types import SearchParams
from ._utilities import _NimbleClientMixin, handle_api_errors


class NimbleSearchToolInput(BaseModel):
    """Input schema for NimbleSearchTool."""

    model_config = {"populate_by_name": True}

    query: str = Field(
        description=(
            "The search query to execute. "
            "Can be natural language questions or keywords."
        )
    )
    max_results: int = Field(
        default=10,
        ge=1,
        le=100,
        alias="num_results",
        description="""Maximum number of search results to return (1-100).

        Actual count may be less depending on available results.
        Default is 10, which balances thoroughness with response time.
        - Use 3-5 for quick lookups or simple questions
        - Use 10-20 for standard research tasks
        - Use 50+ for comprehensive research or when you need diverse sources
        """,
    )
    deep_search: bool = Field(
        default=False,
        description="""Enable deep search mode for comprehensive research.

        **When enabled (True):**
        - Fetches and extracts full page content from each search result
        - Returns rich, detailed information including complete article text
        - Takes longer to execute (typically 5-15 seconds)
        - Ideal for in-depth research, analysis, and detailed answers
        - Provides context needed for comprehensive understanding

        **When disabled (False):**
        - Returns only metadata: title, description, URL, and position
        - Executes quickly (typically 1-3 seconds)
        - Ideal for quick lookups, getting links, or simple fact-checking
        - Requires follow-up extraction if full content is needed

        **Use deep_search=True for:**
        - Research tasks requiring detailed information
        - Analysis or comparison of multiple sources
        - Questions needing full context
        - Content summarization or extraction tasks

        **Use deep_search=False for:**
        - Quick fact checks or simple questions
        - Getting a list of relevant URLs
        - Finding recent news or updates
        - When you plan to extract specific URLs later
        """,
    )
    include_answer: bool = Field(
        default=False,
        description="""Request an LLM-generated answer summary.

        Only available when deep_search=False. When enabled, the API uses
        an LLM to generate a direct answer to your query based on the search
        results. This is useful for getting quick, synthesized answers without
        processing results yourself.

        **Note:** Cannot be used together with deep_search=True.
        """,
    )
    focus: str = Field(
        default="general",
        description="""Search focus mode.

        Available focus modes:
        - "general": Standard web search (default, SERP-based)
        - "news": Real-time news search (SERP-based)
        - "location": Location-based search (SERP-based)
        - "shopping": E-commerce and product search (WSA-based, AI-powered)
        - "geo": Generative engine optimization (WSA-based, AI-powered)
        - "social": Social media content search (WSA-based, AI-powered)
        """,
    )
    time_range: str | None = Field(
        default=None,
        description="""Filter by recency with predefined periods.

        Options:
        - "hour": Content from the last hour
        - "day": Content from the last 24 hours
        - "week": Content from the last 7 days
        - "month": Content from the last 30 days
        - "year": Content from the last year

        This is an alternative to start_date/end_date for quick recency filtering.
        Example: time_range="week" for latest content from past week.
        """,
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="""Whitelist of domains to include in search results.

        Only results from these domains will be returned. Useful for:
        - Searching within specific trusted sources
        - Academic research (e.g., ["edu", "scholar.google.com"])
        - Documentation searches (e.g., ["docs.python.org"])

        Example: ["wikipedia.org", "britannica.com"]
        """,
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description="""Blacklist of domains to exclude from search results.

        Results from these domains will be filtered out. Useful for:
        - Removing low-quality or unreliable sources
        - Avoiding paywalled content
        - Filtering out specific sites

        Example: ["pinterest.com", "facebook.com"]
        """,
    )
    start_date: str | None = Field(
        default=None,
        description="""Filter to only include content published after this date.

        Format: YYYY-MM-DD or YYYY
        Examples: "2024-01-01", "2023"

        Useful for finding recent information or time-sensitive content.
        """,
    )
    end_date: str | None = Field(
        default=None,
        description="""Filter to only include content published before this date.

        Format: YYYY-MM-DD or YYYY
        Examples: "2024-12-31", "2023"

        Useful for historical research or specific time periods.
        """,
    )
    locale: str | None = Field(
        default=None,
        description="""Locale for search results (e.g., 'en', 'fr', 'es').

        Controls the language and regional settings for the search.
        If not specified, defaults to 'en'.
        """,
    )
    country: str | None = Field(
        default=None,
        description="""Country code for search results (e.g., 'US', 'UK', 'FR').

        Controls the regional focus of search results.
        If not specified, defaults to 'US'.
        """,
    )
    output_format: str | None = Field(
        default=None,
        description="""Content output format.

        Available formats:
        - "plain_text": Plain text without formatting
        - "markdown": Markdown-formatted content with structure (default)
        - "simplified_html": Clean HTML without scripts/styles

        If not specified, defaults to 'markdown'.
        """,
    )

    @model_validator(mode="after")
    def validate_deep_search_and_include_answer(self) -> "NimbleSearchToolInput":
        """Validate that deep_search and include_answer are mutually exclusive."""
        if self.deep_search and self.include_answer:
            msg = (
                "deep_search and include_answer cannot both be True. "
                "include_answer is only available when deep_search=False."
            )
            raise ValueError(msg)
        return self


class NimbleSearchTool(_NimbleClientMixin, BaseTool):
    """Search the web using Nimble's Search API.

    This tool provides web search capabilities with optional deep content extraction.
    Results include titles, URLs, snippets, and optionally full page content.

    Use this tool when you need to:
    - Search the web for current information
    - Research topics with comprehensive content
    - Find specific sources or documentation
    - Get up-to-date news or recent developments
    - Gather information from multiple sources

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Base URL for API (defaults to production endpoint).
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
        locale: Locale for results (default: en).
        country: Country code (default: US).
        output_format: Content format - plain_text, markdown (default), simplified_html.
    """

    name: str = "nimble_web_search"
    description: str = (
        "Search the web for current information. Returns search results with "
        "titles, URLs, descriptions, and optionally full page content. Use for "
        "research, fact-checking, finding sources, or gathering information."
    )
    args_schema: type[BaseModel] = NimbleSearchToolInput

    def _build_request_body(
        self,
        query: str,
        max_results: int,
        *,
        deep_search: bool,
        include_answer: bool,
        focus: str,
        time_range: str | None,
        include_domains: list[str] | None,
        exclude_domains: list[str] | None,
        start_date: str | None,
        end_date: str | None,
        locale: str | None,
        country: str | None,
        output_format: str | None,
    ) -> dict[str, Any]:
        """Build search request body."""
        return SearchParams(
            query=query,
            max_results=max_results,
            locale=locale or self.locale,
            country=country or self.country,
            output_format=output_format or self.output_format,
            focus=focus,
            deep_search=deep_search,
            include_answer=include_answer,
            time_range=time_range,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_date=start_date,
            end_date=end_date,
        ).model_dump(exclude_none=True, by_alias=True)

    def _run(
        self,
        query: str,
        max_results: int = 10,
        *,
        deep_search: bool = False,
        include_answer: bool = False,
        focus: str = "general",
        time_range: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        locale: str | None = None,
        country: str | None = None,
        output_format: str | None = None,
    ) -> dict[str, Any]:
        """Execute search synchronously."""
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        request_body = self._build_request_body(
            query=query,
            max_results=max_results,
            deep_search=deep_search,
            include_answer=include_answer,
            focus=focus,
            time_range=time_range,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_date=start_date,
            end_date=end_date,
            locale=locale,
            country=country,
            output_format=output_format,
        )

        with handle_api_errors(operation="search"):
            response = self._sync_client.post("/search", json=request_body)
            response.raise_for_status()
            return response.json()

    async def _arun(
        self,
        query: str,
        max_results: int = 10,
        *,
        deep_search: bool = False,
        include_answer: bool = False,
        focus: str = "general",
        time_range: str | None = None,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        locale: str | None = None,
        country: str | None = None,
        output_format: str | None = None,
    ) -> dict[str, Any]:
        """Execute search asynchronously."""
        if self._async_client is None:
            msg = "Async client not initialized"
            raise RuntimeError(msg)

        request_body = self._build_request_body(
            query=query,
            max_results=max_results,
            deep_search=deep_search,
            include_answer=include_answer,
            focus=focus,
            time_range=time_range,
            include_domains=include_domains,
            exclude_domains=exclude_domains,
            start_date=start_date,
            end_date=end_date,
            locale=locale,
            country=country,
            output_format=output_format,
        )

        with handle_api_errors(operation="search"):
            response = await self._async_client.post("/search", json=request_body)
            response.raise_for_status()
            return response.json()
