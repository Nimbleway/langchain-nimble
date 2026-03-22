"""Nimble Search API retriever implementations."""

from typing import Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents.base import Document
from langchain_core.retrievers import BaseRetriever
from nimble_python.types import ExtractResponse, SearchResponse
from nimble_python.types.search_response import (
    Result,
    ResultMetadataSerpMetadata,
)
from pydantic import Field

from ._types import OUTPUT_FORMAT_TO_SDK_FORMATS, BrowserlessDriver
from ._utilities import _NimbleClientMixin, handle_api_errors


def _search_result_to_document(result: Result) -> Document:
    """Convert a single SDK search result to a LangChain Document."""
    meta = result.metadata
    position = -1
    entity_type = ""
    if isinstance(meta, ResultMetadataSerpMetadata):
        position = meta.position
        entity_type = meta.entity_type

    return Document(
        page_content=result.content or "",
        metadata={
            "title": result.title or "",
            "description": result.description or "",
            "url": result.url or "",
            "position": position,
            "entity_type": entity_type,
        },
    )


def _parse_search_response(
    response: SearchResponse,
) -> list[Document]:
    """Parse SDK SearchResponse into LangChain Documents."""
    return [_search_result_to_document(r) for r in (response.results or [])]


def _parse_extract_response(
    response: ExtractResponse,
) -> list[Document]:
    """Parse SDK ExtractResponse into a single-item Document list."""
    content = ""
    if response.data and response.data.markdown:
        content = response.data.markdown

    return [
        Document(
            page_content=content,
            metadata={
                "title": "",
                "description": "",
                "url": response.url or "",
                "position": 0,
                "entity_type": "",
            },
        )
    ]


class NimbleSearchRetriever(_NimbleClientMixin, BaseRetriever):
    """Search retriever for Nimble API.

    Retrieves search results with full page content extraction.
    Supports SERP focuses (general, news, location) and WSA focuses
    (shopping, geo, social).

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_results: Maximum number of results to return (1-100, default: 3).
            Alias: k.
        focus: Search focus mode - general, news, location,
            shopping, geo, social.
        search_depth: Search depth - lite (metadata only, default),
            fast (rich content at low latency, Enterprise only),
            deep (full content).
        include_answer: Generate LLM answer summary.
        include_domains: Whitelist of domains to include.
        exclude_domains: Blacklist of domains to exclude.
        time_range: Filter by recency - hour, day, week, month, year.
        start_date: Filter results after date (YYYY-MM-DD or YYYY).
        end_date: Filter results before date (YYYY-MM-DD or YYYY).
        locale: Locale for results (default: en).
        country: Country code (default: US).
        output_format: Content format - plain_text, markdown (default),
            simplified_html.
    """

    max_results: int = Field(default=3, ge=1, le=100, alias="k")
    focus: str = "general"
    search_depth: str = "lite"
    include_answer: bool = False
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    time_range: str | None = None
    start_date: str | None = None
    end_date: str | None = None

    def _build_search_kwargs(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Build keyword arguments for SDK search() call."""
        search_kwargs: dict[str, Any] = {
            "query": query,
            "max_results": kwargs.get("max_results", kwargs.get("k", self.max_results)),
            "locale": kwargs.get("locale", self.locale),
            "country": kwargs.get("country", self.country),
            "output_format": kwargs.get("output_format", self.output_format),
            "focus": kwargs.get("focus", self.focus),
            "search_depth": kwargs.get("search_depth", self.search_depth),
        }

        include_answer = kwargs.get("include_answer", self.include_answer)
        if include_answer:
            search_kwargs["include_answer"] = include_answer

        optional_fields = (
            "include_domains",
            "exclude_domains",
            "time_range",
            "start_date",
            "end_date",
        )
        for field in optional_fields:
            val = getattr(self, field)
            if val is not None:
                search_kwargs[field] = val

        return search_kwargs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        with handle_api_errors(operation="search"):
            search_kwargs = self._build_search_kwargs(query, **kwargs)
            response = self._sync_client.search(**search_kwargs)
            return _parse_search_response(response)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        if self._async_client is None:
            msg = "Async client not initialized"
            raise RuntimeError(msg)

        with handle_api_errors(operation="search"):
            response = await self._async_client.search(
                **self._build_search_kwargs(query, **kwargs)
            )
            return _parse_search_response(response)


class NimbleExtractRetriever(_NimbleClientMixin, BaseRetriever):
    """Extract retriever for Nimble API.

    Extracts content from a single URL passed via the query parameter.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        locale: Locale for results (default: en).
        country: Country code (default: US).
        output_format: Content format - plain_text, markdown (default),
            simplified_html.
        driver: Browser driver to use (vx6, vx8, vx8-pro, vx10, vx10-pro,
            vx12, vx12-pro). If not specified, API selects the most
            appropriate driver.
        wait: Optional delay in milliseconds for render flow.

    Example:
        >>> retriever = NimbleExtractRetriever()
        >>> docs = await retriever.ainvoke("https://example.com")
    """

    driver: BrowserlessDriver | None = None
    wait: int | None = None

    def _build_extract_kwargs(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Build keyword arguments for SDK extract() call."""
        fmt = kwargs.get("output_format", self.output_format)
        extract_kwargs: dict[str, Any] = {
            "url": query,
            "locale": kwargs.get("locale", self.locale),
            "country": kwargs.get("country", self.country),
            "formats": OUTPUT_FORMAT_TO_SDK_FORMATS.get(fmt, ["markdown"]),
        }

        driver = kwargs.get("driver", self.driver)
        if driver is not None:
            extract_kwargs["driver"] = driver.value

        wait = kwargs.get("wait", self.wait)
        if wait is not None:
            extract_kwargs["request_timeout"] = float(wait)

        return extract_kwargs

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        with handle_api_errors(operation="extract"):
            extract_kwargs = self._build_extract_kwargs(query, **kwargs)
            response = self._sync_client.extract(**extract_kwargs)
            return _parse_extract_response(response)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
        **kwargs: Any,
    ) -> list[Document]:
        if self._async_client is None:
            msg = "Async client not initialized"
            raise RuntimeError(msg)

        with handle_api_errors(operation="extract"):
            response = await self._async_client.extract(
                **self._build_extract_kwargs(query, **kwargs)
            )
            return _parse_extract_response(response)
