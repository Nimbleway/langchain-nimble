"""Shared type definitions for Nimble Search API."""

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class SearchFocus(str, Enum):
    """Search focus mode/specialization."""

    GENERAL = "general"
    NEWS = "news"
    LOCATION = "location"
    SHOPPING = "shopping"
    GEO = "geo"
    SOCIAL = "social"


class OutputFormat(str, Enum):
    """Content output format."""

    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    SIMPLIFIED_HTML = "simplified_html"


class BrowserlessDriver(str, Enum):
    """Browserless drivers available for web extraction."""

    VX6 = "vx6"
    VX8 = "vx8"
    VX8_PRO = "vx8-pro"
    VX10 = "vx10"
    VX10_PRO = "vx10-pro"
    VX12 = "vx12"
    VX12_PRO = "vx12-pro"


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
    output_format: OutputFormat = Field(
        default=OutputFormat.MARKDOWN,
        description="Format for output content (default: markdown)",
    )


class SearchParams(BaseParams):
    """Search parameters for /search endpoint."""

    model_config = {"populate_by_name": True}

    query: str = Field(
        description="Search query string",
    )
    max_results: int = Field(
        default=3,
        ge=1,
        le=100,
        alias="num_results",
        description=(
            "Maximum number of results to return (1-100). Actual count may be less."
        ),
    )
    focus: SearchFocus = Field(
        default=SearchFocus.GENERAL,
        description=(
            "Search focus mode. "
            "Options: general (SERP), news (SERP), location (SERP), "
            "shopping (WSA), geo (WSA), social (WSA)"
        ),
    )
    deep_search: bool = Field(
        default=True,
        description=(
            "Enable deep search to fetch full page content. "
            "When False, returns only metadata (title, description, URL)."
        ),
    )
    include_answer: bool = Field(
        default=False,
        description=(
            "Generate LLM answer summary (only available when deep_search=False)."
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
    time_range: str | None = Field(
        default=None,
        description=(
            "Filter by recency with predefined periods. "
            "Options: hour, day, week, month, year. "
            "Alternative to start_date/end_date for quick recency filtering."
        ),
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
    driver: BrowserlessDriver | None = Field(
        default=None,
        description=(
            "Browserless driver to use for extraction: "
            "vx6, vx8, vx8-pro, vx10, vx10-pro, vx12, vx12-pro"
        ),
    )
    wait: int | None = Field(
        default=None,
        description="Optional delay in milliseconds for render flow",
    )
