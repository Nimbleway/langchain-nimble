"""LangChain extract tool for Nimble Content Extraction API."""

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ._types import BrowserlessDriver, ExtractParams
from ._utilities import _NimbleClientMixin, handle_api_errors


class NimbleExtractToolInput(BaseModel):
    """Input schema for NimbleExtractTool."""

    urls: list[str] = Field(
        min_length=1,
        max_length=20,
        description="""List of URLs to extract content from (1-20 URLs).

        Provide one or more URLs to extract their full page content.
        The tool will fetch and parse each URL, returning structured content.

        **Examples:**
        - Single URL: ["https://example.com/article"]
        - Multiple URLs: ["https://site1.com", "https://site2.com", "https://site3.com"]

        **Best practices:**
        - Use this tool when you have specific URLs to extract
        - For general web research, use the search tool first
        - Batch multiple URLs together for efficiency (up to 20)
        - URLs must be well-formed and accessible
        """,
    )
    driver: BrowserlessDriver | None = Field(
        default=None,
        description="""Browser driver technology for content extraction.

        Available drivers (Browserless Driver Technology):
        - "vx6": Fast native requests - standard websites, static content
        - "vx8": Enhanced rendering - JavaScript-heavy sites, SPAs
        - "vx8-pro": Enhanced rendering with extended capabilities
        - "vx10": Advanced rendering - maximum compatibility for complex sites
        - "vx10-pro": Advanced rendering with extended capabilities
        - "vx12": Premium rendering for the most demanding sites
        - "vx12-pro": Premium rendering with extended capabilities

        If not specified, the API will select the most appropriate driver.
        Higher driver levels have different pricing.
        """,
    )
    wait: int | None = Field(
        default=None,
        description="""Optional delay in milliseconds before content extraction.

        Use this when pages have dynamic content that loads after initial render:
        - None: Extract immediately after page loads (default, fastest)
        - 1000-5000: Wait 1-5 seconds for most dynamic content
        - 5000+: Wait longer for very slow-loading pages

        **When to use wait:**
        - Pages with lazy-loaded content
        - Single-page applications (SPAs) that render client-side
        - Pages with animations or transitions
        - Content behind async data fetching

        **When NOT to use wait:**
        - Static HTML pages (no benefit, just slower)
        - Server-rendered content (already loaded)
        - When speed is critical and partial content is acceptable

        Trade-off: Longer wait times = more complete content but slower extraction.
        """,
    )
    locale: str | None = Field(
        default=None,
        description="""Locale for content extraction (e.g., 'en', 'fr', 'es').

        Controls the language context for extraction.
        If not provided, uses the tool's configured locale (default: 'en').
        """,
    )
    country: str | None = Field(
        default=None,
        description="""Country code for content extraction (e.g., 'US', 'UK', 'FR').

        Controls the regional context for extraction.
        If not provided, uses the tool's configured country (default: 'US').
        """,
    )
    output_format: str | None = Field(
        default=None,
        description="""Content output format.

        Available formats:
        - "plain_text": Plain text without formatting
        - "markdown": Markdown-formatted content with structure (default)
        - "simplified_html": Clean HTML without scripts/styles

        If not provided, uses the tool's configured output format
        (default: 'markdown').
        """,
    )


class NimbleExtractTool(_NimbleClientMixin, BaseTool):
    """Extract content from specific URLs using Nimble's Extract API.

    This tool fetches and extracts full page content from provided URLs.
    Results include structured content in your chosen format
    (markdown, plain text, or HTML).

    Use this tool when you need to:
    - Extract full content from specific URLs
    - Get detailed page content for analysis or summarization
    - Fetch content from multiple URLs in batch
    - Access dynamically-loaded content from JavaScript-heavy sites

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Base URL for API (defaults to production endpoint).
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
        locale: Locale for results (default: en).
        country: Country code (default: US).
        output_format: Content format - plain_text, markdown (default), simplified_html.

    Note:
        driver and wait parameters are configured per-request via tool input,
        not at tool initialization.
    """

    name: str = "nimble_extract_content"
    description: str = (
        "Extract full page content from specific URLs. Returns structured content "
        "from one or more URLs (up to 20). Use when you have specific URLs to "
        "extract, or after using search to find relevant pages."
    )
    args_schema: type[BaseModel] = NimbleExtractToolInput

    def _build_request_body(
        self,
        urls: list[str],
        *,
        driver: BrowserlessDriver | None,
        wait: int | None,
        locale: str | None,
        country: str | None,
        output_format: str | None,
    ) -> dict[str, Any]:
        return ExtractParams(
            links=urls,
            locale=locale or self.locale,
            country=country or self.country,
            output_format=output_format or self.output_format,
            driver=driver,
            wait=wait,
        ).model_dump(exclude_none=True)

    def _run(
        self,
        urls: list[str],
        *,
        driver: BrowserlessDriver | None = None,
        wait: int | None = None,
        locale: str | None = None,
        country: str | None = None,
        output_format: str | None = None,
    ) -> dict[str, Any]:
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        request_body = self._build_request_body(
            urls=urls,
            driver=driver,
            wait=wait,
            locale=locale,
            country=country,
            output_format=output_format,
        )

        with handle_api_errors(operation="extract"):
            response = self._sync_client.post("/extract", json=request_body)
            response.raise_for_status()
            return response.json()

    async def _arun(
        self,
        urls: list[str],
        *,
        driver: BrowserlessDriver | None = None,
        wait: int | None = None,
        locale: str | None = None,
        country: str | None = None,
        output_format: str | None = None,
    ) -> dict[str, Any]:
        if self._async_client is None:
            msg = "Async client not initialized"
            raise RuntimeError(msg)

        request_body = self._build_request_body(
            urls=urls,
            driver=driver,
            wait=wait,
            locale=locale,
            country=country,
            output_format=output_format,
        )

        with handle_api_errors(operation="extract"):
            response = await self._async_client.post("/extract", json=request_body)
            response.raise_for_status()
            return response.json()
