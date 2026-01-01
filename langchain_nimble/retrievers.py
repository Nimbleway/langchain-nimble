"""Nimble Search API retriever implementations."""

from abc import abstractmethod
from typing import Any

from langchain_core.callbacks.manager import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents.base import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field

from ._types import BrowserlessDriver, ExtractParams, SearchParams
from ._utilities import _NimbleClientMixin, handle_api_errors


class _NimbleBaseRetriever(_NimbleClientMixin, BaseRetriever):
    """Base retriever with shared API client logic."""

    @abstractmethod
    def _build_request_body(self, query: str, **kwargs: Any) -> dict[str, Any]:
        """Build API request body."""
        ...

    @abstractmethod
    def _get_endpoint(self) -> str:
        """Return API endpoint path."""
        ...

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        with handle_api_errors(operation=f"{self._get_endpoint()} request"):
            response = self._sync_client.post(
                self._get_endpoint(), json=self._build_request_body(query, **kwargs)
            )
            response.raise_for_status()
            return self._parse_response(response.json())

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

        with handle_api_errors(operation=f"{self._get_endpoint()} request"):
            response = await self._async_client.post(
                self._get_endpoint(), json=self._build_request_body(query, **kwargs)
            )
            response.raise_for_status()
            return self._parse_response(response.json())

    def _parse_response(self, raw_json_content: dict[str, Any]) -> list[Document]:
        """Parse API response into Documents.

        Response structure:
        - results: list of result objects
        - content: main content of each result
        - title, url, description: top-level fields
        - metadata: nested object with position, entity_type, etc.
        """
        return [
            Document(
                page_content=doc.get("content", ""),
                metadata={
                    "title": doc.get("title", ""),
                    "description": doc.get("description", ""),
                    "url": doc.get("url", ""),
                    "position": doc.get("metadata", {}).get("position", -1),
                    "entity_type": doc.get("metadata", {}).get("entity_type", ""),
                },
            )
            for doc in raw_json_content.get("results", [])
        ]


class NimbleSearchRetriever(_NimbleBaseRetriever):
    """Search retriever for Nimble API.

    Retrieves search results with full page content extraction.
    Supports SERP focuses (general, news, location) and WSA focuses
    (shopping, geo, social).

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Base URL for API (defaults to production endpoint).
        max_results: Maximum number of results to return (1-100, default: 3). Alias: k.
        focus: Search focus mode - general, news, location,
            shopping, geo, social.
        deep_search: Fetch full page content (default: True).
        include_answer: Generate LLM answer (only with deep_search=False).
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
    deep_search: bool = True
    include_answer: bool = False
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    time_range: str | None = None
    start_date: str | None = None
    end_date: str | None = None

    def _get_endpoint(self) -> str:
        return "/search"

    def _build_request_body(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return SearchParams(
            query=query,
            max_results=kwargs.get("max_results", kwargs.get("k", self.max_results)),
            locale=kwargs.get("locale", self.locale),
            country=kwargs.get("country", self.country),
            output_format=kwargs.get("output_format", self.output_format),
            focus=kwargs.get("focus", self.focus),
            deep_search=kwargs.get("deep_search", self.deep_search),
            include_answer=kwargs.get("include_answer", self.include_answer),
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            time_range=self.time_range,
            start_date=self.start_date,
            end_date=self.end_date,
        ).model_dump(exclude_none=True, by_alias=True)


class NimbleExtractRetriever(_NimbleBaseRetriever):
    """Extract retriever for Nimble API.

    Extracts content from a single URL passed via the query parameter.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Base URL for API (defaults to production endpoint).
        locale: Locale for results (default: en).
        country: Country code (default: US).
        output_format: Content format - plain_text, markdown (default),
            simplified_html.
        driver: Browser driver to use (vx6, vx8, vx8-pro, vx10, vx10-pro, vx12,
            vx12-pro). If not specified, API selects the most appropriate driver.
        wait: Optional delay in milliseconds for render flow.

    Example:
        >>> retriever = NimbleExtractRetriever()
        >>> docs = await retriever.ainvoke("https://example.com")
    """

    driver: BrowserlessDriver | None = None
    wait: int | None = None

    def _get_endpoint(self) -> str:
        return "/extract"

    def _build_request_body(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return ExtractParams(
            links=[query],
            locale=kwargs.get("locale", self.locale),
            country=kwargs.get("country", self.country),
            output_format=kwargs.get("output_format", self.output_format),
            driver=kwargs.get("driver", self.driver),
            wait=self.wait,
        ).model_dump(exclude_none=True)
