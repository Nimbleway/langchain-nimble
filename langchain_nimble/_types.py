"""Shared type definitions for Nimble Search API."""

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class SearchTopic(str, Enum):
    """Search topic/specialization."""

    GENERAL = "general"
    NEWS = "news"
    LOCATION = "location"


class ParsingType(str, Enum):
    """Content parsing format."""

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
    """Search parameters for /search endpoint."""

    query: str = Field(
        description="Search query string",
    )
    num_results: int = Field(
        default=3,
        ge=1,
        le=100,
        description="Number of results to return (1-100)",
    )
    topic: SearchTopic = Field(
        default=SearchTopic.GENERAL,
        description="Search topic/specialization (general, news, or location)",
    )
    deep_search: bool = Field(
        default=True,
        description=(
            "Enable deep search to fetch full page content. "
            "When False, returns only metadata (title, snippet, URL)."
        ),
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

    @model_validator(mode="after")
    def _validate_logic(self) -> "SearchParams":
        if self.deep_search and self.include_answer:
            msg = "`include_answer` cannot be True when `deep_search` is True."
            raise ValueError(msg)
        return self


class ExtractParams(BaseParams):
    """Extract parameters for /extract endpoint."""

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
