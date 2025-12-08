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

from ._types import ExtractParams, SearchParams
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
        """Parse API response into Documents."""
        return [
            Document(
                page_content=doc.get("page_content", ""),
                metadata={
                    "title": doc.get("metadata", {}).get("title", ""),
                    "snippet": doc.get("metadata", {}).get("snippet", ""),
                    "url": doc.get("metadata", {}).get("url", ""),
                    "position": doc.get("metadata", {}).get("position", -1),
                    "entity_type": doc.get("metadata", {}).get("entity_type", ""),
                },
            )
            for doc in raw_json_content.get("body", [])
        ]


class NimbleSearchRetriever(_NimbleBaseRetriever):
    """Search retriever for Nimble API.

    Retrieves search results with full page content extraction.
    Supports general, news, and location topics.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Base URL for API (defaults to production endpoint).
        num_results: Number of results to return (1-100, default: 3). Alias: k.
        topic: Search topic - general, news, or location (default: general).
        deep_search: Fetch full page content (default: True).
        include_answer: Generate LLM answer (only with deep_search=False).
        include_domains: Whitelist of domains to include.
        exclude_domains: Blacklist of domains to exclude.
        start_date: Filter results after date (YYYY-MM-DD or YYYY).
        end_date: Filter results before date (YYYY-MM-DD or YYYY).
        locale: Locale for results (default: en).
        country: Country code (default: US).
        parsing_type: Content format - plain_text, markdown (default), simplified_html.
    """

    num_results: int = Field(default=3, ge=1, le=100, alias="k")
    topic: str = "general"
    deep_search: bool = True
    include_answer: bool = False
    include_domains: list[str] | None = None
    exclude_domains: list[str] | None = None
    start_date: str | None = None
    end_date: str | None = None

    def _get_endpoint(self) -> str:
        return "/search"

    def _build_request_body(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return SearchParams(
            query=query,
            num_results=kwargs.get("num_results", kwargs.get("k", self.num_results)),
            locale=kwargs.get("locale", self.locale),
            country=kwargs.get("country", self.country),
            parsing_type=kwargs.get("parsing_type", self.parsing_type),
            topic=kwargs.get("topic", self.topic),
            deep_search=kwargs.get("deep_search", self.deep_search),
            include_answer=kwargs.get("include_answer", self.include_answer),
            include_domains=self.include_domains,
            exclude_domains=self.exclude_domains,
            start_date=self.start_date,
            end_date=self.end_date,
        ).model_dump(exclude_none=True)


class NimbleExtractRetriever(_NimbleBaseRetriever):
    """Extract retriever for Nimble API.

    Extracts content from a single URL passed via the query parameter.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Base URL for API (defaults to production endpoint).
        locale: Locale for results (default: en).
        country: Country code (default: US).
        parsing_type: Content format - plain_text, markdown (default), simplified_html.
        driver: Browser driver to use (default: vx6).
        wait: Optional delay in milliseconds for render flow.

    Example:
        >>> retriever = NimbleExtractRetriever()
        >>> docs = await retriever.ainvoke("https://example.com")
    """

    driver: str = "vx6"
    wait: int | None = None

    def _get_endpoint(self) -> str:
        return "/extract"

    def _build_request_body(self, query: str, **kwargs: Any) -> dict[str, Any]:
        return ExtractParams(
            links=[query],
            locale=kwargs.get("locale", self.locale),
            country=kwargs.get("country", self.country),
            parsing_type=kwargs.get("parsing_type", self.parsing_type),
            driver=kwargs.get("driver", self.driver),
            wait=self.wait,
        ).model_dump(exclude_none=True)
