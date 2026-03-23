"""LangChain tool for Nimble Crawl API."""

from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.tools import BaseTool, ToolException
from pydantic import BaseModel, Field

from langchain_nimble._utilities import _NimbleClientMixin, handle_api_errors

_TERMINAL_STATUSES = frozenset({"succeeded", "failed", "canceled"})


class NimbleCrawlToolInput(BaseModel):
    """Input schema for NimbleCrawlTool."""

    url: str = Field(
        description="""The URL to start crawling from.

        The crawler will discover and extract content from pages
        starting at this URL, following links based on the configured
        depth and path filters.
        """,
    )
    limit: int | None = Field(
        default=None,
        description="Maximum number of pages to crawl.",
    )
    max_discovery_depth: int | None = Field(
        default=None,
        description="Maximum link-following depth from the start URL.",
    )
    allow_external_links: bool | None = Field(
        default=None,
        description="Whether to follow links to external domains.",
    )
    allow_subdomains: bool | None = Field(
        default=None,
        description="Whether to follow links to subdomains.",
    )
    crawl_entire_domain: bool | None = Field(
        default=None,
        description="Whether to crawl the entire domain.",
    )
    include_paths: list[str] | None = Field(
        default=None,
        description="""URL path patterns to include.

        Only pages matching these patterns will be crawled.
        Example: ["/blog/*", "/docs/*"]
        """,
    )
    exclude_paths: list[str] | None = Field(
        default=None,
        description="""URL path patterns to exclude.

        Pages matching these patterns will be skipped.
        Example: ["/admin/*", "/private/*"]
        """,
    )
    sitemap: str | None = Field(
        default=None,
        description="""Sitemap handling strategy.

        Options:
        - "skip": Ignore sitemaps
        - "include": Use sitemaps alongside crawling
        - "only": Only use sitemaps for URL discovery
        """,
    )
    name: str | None = Field(
        default=None,
        description="Optional name for the crawl job.",
    )


class NimbleCrawlTool(_NimbleClientMixin, BaseTool):
    """Crawl a website using Nimble's Crawl API.

    Starts an asynchronous crawl job and polls for completion. Returns the
    list of crawled page tasks with their extracted content.

    The crawl job runs server-side. This tool polls the job status at
    regular intervals until it completes, fails, or times out.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
        locale: Locale for results (default: en).
        country: Country code (default: US).
        polling_interval: Seconds between status polls (default: 5.0).
        timeout: Maximum seconds to wait for completion (default: 300.0).
    """

    name: str = "nimble_crawl"
    description: str = (
        "Crawl a website to extract content from multiple pages. "
        "Discovers and extracts pages starting from a URL, following "
        "links based on depth and path filters. Use for bulk content "
        "extraction from websites."
    )
    args_schema: type[BaseModel] = NimbleCrawlToolInput
    handle_tool_error: bool = True

    polling_interval: float = Field(
        default=5.0,
        gt=0,
        description="Seconds between status polls.",
    )
    timeout: float = Field(
        default=300.0,
        gt=0,
        description="Maximum seconds to wait for crawl completion.",
    )

    def _build_crawl_kwargs(
        self,
        url: str,
        *,
        limit: int | None,
        max_discovery_depth: int | None,
        allow_external_links: bool | None,
        allow_subdomains: bool | None,
        crawl_entire_domain: bool | None,
        include_paths: list[str] | None,
        exclude_paths: list[str] | None,
        sitemap: str | None,
        name: str | None,
    ) -> dict[str, Any]:
        """Build keyword arguments for SDK crawl.run() call."""
        kwargs: dict[str, Any] = {"url": url}

        optional: dict[str, Any] = {
            "limit": limit,
            "max_discovery_depth": max_discovery_depth,
            "allow_external_links": allow_external_links,
            "allow_subdomains": allow_subdomains,
            "crawl_entire_domain": crawl_entire_domain,
            "include_paths": include_paths,
            "exclude_paths": exclude_paths,
            "sitemap": sitemap,
            "name": name,
        }

        kwargs.update({k: v for k, v in optional.items() if v is not None})

        return kwargs

    def _check_status(self, crawl_id: str, response: Any) -> list[Any] | None:
        """Check crawl status response. Returns tasks on success, None to continue."""
        if response.status == "succeeded":
            return [t.model_dump() for t in (response.tasks or [])]
        if response.status in _TERMINAL_STATUSES:
            msg = f"Crawl {crawl_id} ended with status '{response.status}'"
            raise ToolException(msg)
        return None

    def _poll_sync(self, crawl_id: str) -> list[Any]:
        """Poll crawl status synchronously until terminal state."""
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        deadline = time.monotonic() + self.timeout
        first = True
        while True:
            if not first:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                time.sleep(min(self.polling_interval, remaining))
            first = False

            with handle_api_errors(operation="crawl status"):
                response = self._sync_client.crawl.status(crawl_id)

            result = self._check_status(crawl_id, response)
            if result is not None:
                return result

            if time.monotonic() >= deadline:
                break

        msg = f"Crawl {crawl_id} timed out after {self.timeout}s"
        raise ToolException(msg)

    async def _poll_async(self, crawl_id: str) -> list[Any]:
        """Poll crawl status asynchronously until terminal state."""
        if self._async_client is None:
            msg = "Async client not initialized"
            raise RuntimeError(msg)

        deadline = time.monotonic() + self.timeout
        first = True
        while True:
            if not first:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(self.polling_interval, remaining))
            first = False

            with handle_api_errors(operation="crawl status"):
                response = await self._async_client.crawl.status(crawl_id)

            result = self._check_status(crawl_id, response)
            if result is not None:
                return result

            if time.monotonic() >= deadline:
                break

        msg = f"Crawl {crawl_id} timed out after {self.timeout}s"
        raise ToolException(msg)

    def _run(
        self,
        url: str,
        *,
        limit: int | None = None,
        max_discovery_depth: int | None = None,
        allow_external_links: bool | None = None,
        allow_subdomains: bool | None = None,
        crawl_entire_domain: bool | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        sitemap: str | None = None,
        name: str | None = None,
    ) -> list[Any]:
        """Execute crawl synchronously."""
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        crawl_kwargs = self._build_crawl_kwargs(
            url=url,
            limit=limit,
            max_discovery_depth=max_discovery_depth,
            allow_external_links=allow_external_links,
            allow_subdomains=allow_subdomains,
            crawl_entire_domain=crawl_entire_domain,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            sitemap=sitemap,
            name=name,
        )

        with handle_api_errors(operation="crawl"):
            response = self._sync_client.crawl.run(**crawl_kwargs)

        return self._poll_sync(response.crawl_id)

    async def _arun(
        self,
        url: str,
        *,
        limit: int | None = None,
        max_discovery_depth: int | None = None,
        allow_external_links: bool | None = None,
        allow_subdomains: bool | None = None,
        crawl_entire_domain: bool | None = None,
        include_paths: list[str] | None = None,
        exclude_paths: list[str] | None = None,
        sitemap: str | None = None,
        name: str | None = None,
    ) -> list[Any]:
        """Execute crawl asynchronously."""
        if self._async_client is None:
            msg = "Async client not initialized"
            raise RuntimeError(msg)

        crawl_kwargs = self._build_crawl_kwargs(
            url=url,
            limit=limit,
            max_discovery_depth=max_discovery_depth,
            allow_external_links=allow_external_links,
            allow_subdomains=allow_subdomains,
            crawl_entire_domain=crawl_entire_domain,
            include_paths=include_paths,
            exclude_paths=exclude_paths,
            sitemap=sitemap,
            name=name,
        )

        with handle_api_errors(operation="crawl"):
            response = await self._async_client.crawl.run(**crawl_kwargs)

        return await self._poll_async(response.crawl_id)
