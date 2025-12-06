"""Shared type definitions for Nimble Search API."""

from enum import Enum

from pydantic import BaseModel, Field


class SearchEngine(str, Enum):
    """Enum representing the search engines supported by Nimble.

    ⚠️ DEPRECATED: This parameter is ignored. Use 'topic' parameter instead.
    """

    GOOGLE = "google_search"
    GOOGLE_SGE = "google_sge"
    BING = "bing_search"
    YANDEX = "yandex_search"


class SearchTopic(str, Enum):
    """Enum representing the search topic/specialization.

    Controls which search engine and parameters are used internally.
    """

    GENERAL = "general"  # Default - broad web search
    NEWS = "news"  # Real-time news search
    LOCATION = "location"  # Location-based search


class ParsingType(str, Enum):
    """Enum representing the parsing types supported by Nimble."""

    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    SIMPLIFIED_HTML = "simplified_html"


class BaseParams(BaseModel):
    """Base parameters shared by search and extract endpoints."""

    locale: str = Field(
        default="en",
        description="Locale for results (e.g., 'en', 'fr', 'es')",
    )
    country: str = Field(
        default="US",
        description="Country code for results (e.g., 'US', 'UK', 'FR')",
    )
    parsing_type: ParsingType = Field(
        default=ParsingType.PLAIN_TEXT,
        description="Format for parsing result content",
    )


class SearchParams(BaseParams):
    """Search parameters for Nimble Search API /search endpoint.

    This model provides parameters for the search retriever and tool.
    The API will validate all constraints server-side.
    """

    query: str = Field(
        description="Search query string",
    )
    num_results: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of results to return (1-100)",
    )
    search_engine: SearchEngine | None = Field(
        default=None,
        description=(
            "⚠️ DEPRECATED: This parameter is ignored. Use 'topic' parameter instead."
        ),
    )
    topic: SearchTopic = Field(
        default=SearchTopic.GENERAL,
        description="Search topic/specialization (general, news, or location)",
    )
    deep_search: bool = Field(
        default=True,
        description="""Enable deep search mode for comprehensive research.

        When True:
        - Fetches and extracts full page content for each search result
        - Provides detailed information, not just snippets
        - Takes longer but returns richer data

        When False:
        - Returns only metadata (title, snippet, URL)
        - Faster response times
        - Good for quick lookups

        Use deep_search=True for: In-depth research, analysis, comprehensive answers
        Use deep_search=False for: Quick facts, simple lookups, when speed matters
        """,
    )
    include_answer: bool = Field(
        default=False,
        description=(
            "Generate LLM answer summary (only available when deep_search=False)"
        ),
    )
    include_domains: list[str] | None = Field(
        default=None,
        description="List of domains to include in search results",
    )
    exclude_domains: list[str] | None = Field(
        default=None,
        description="List of domains to exclude from search results",
    )
    start_date: str | None = Field(
        default=None,
        description="Filter results after this date (format: YYYY-MM-DD or YYYY)",
    )
    end_date: str | None = Field(
        default=None,
        description="Filter results before this date (format: YYYY-MM-DD or YYYY)",
    )


class ExtractParams(BaseParams):
    """Extract parameters for Nimble Search API /extract endpoint.

    This model provides parameters for the extract retriever functionality.
    The API will validate all constraints server-side.
    """

    links: list[str] = Field(
        min_length=1,
        max_length=20,
        description="List of URLs to extract content from (1-20 URLs)",
    )
    driver: str = Field(
        default="vx6",
        description="Browser driver to use for extraction",
    )
    wait: int | None = Field(
        default=None,
        description="Optional delay in milliseconds for render flow",
    )
