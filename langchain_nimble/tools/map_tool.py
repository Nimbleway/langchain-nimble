"""LangChain tool for Nimble Map API."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from langchain_nimble._utilities import _NimbleClientMixin, handle_api_errors


class NimbleMapToolInput(BaseModel):
    """Input schema for NimbleMapTool."""

    url: str = Field(
        description="""The URL of the website to map.

        Discovers all links on the given domain. Useful for understanding
        site structure before crawling or extracting specific pages.
        """,
    )
    limit: int | None = Field(
        default=None,
        description="""Maximum number of links to return.

        If not specified, returns all discovered links.
        """,
    )
    domain_filter: str | None = Field(
        default=None,
        description="""Filter links by domain scope.

        Options:
        - "domain": Only links on the same domain
        - "subdomain": Include subdomain links
        - "all": All discovered links including external
        """,
    )
    sitemap: str | None = Field(
        default=None,
        description="""Sitemap handling strategy.

        Options:
        - "skip": Ignore sitemaps
        - "include": Use sitemaps alongside crawling
        - "only": Only use sitemaps for discovery
        """,
    )
    locale: str | None = Field(
        default=None,
        description="Override locale for the request (e.g., 'en', 'fr').",
    )
    country: str | None = Field(
        default=None,
        description="Override country code for the request (e.g., 'US', 'UK').",
    )


class NimbleMapTool(_NimbleClientMixin, BaseTool):
    """Discover all links on a website using Nimble's Map API.

    Returns a list of URLs with titles and descriptions found on the domain.
    Use this tool to understand site structure before crawling or extracting.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
        locale: Locale for results (default: en).
        country: Country code (default: US).
    """

    name: str = "nimble_map"
    description: str = (
        "Discover all links on a website. Returns URLs with titles and "
        "descriptions. Use to understand site structure before crawling "
        "or extracting specific pages."
    )
    args_schema: type[BaseModel] = NimbleMapToolInput
    handle_tool_error: bool = True

    def _build_map_kwargs(
        self,
        url: str,
        *,
        limit: int | None,
        domain_filter: str | None,
        sitemap: str | None,
        locale: str | None,
        country: str | None,
    ) -> dict[str, Any]:
        """Build keyword arguments for SDK map() call."""
        kwargs: dict[str, Any] = {
            "url": url,
            "locale": locale or self.locale,
            "country": country or self.country,
        }

        if limit is not None:
            kwargs["limit"] = limit
        if domain_filter is not None:
            kwargs["domain_filter"] = domain_filter
        if sitemap is not None:
            kwargs["sitemap"] = sitemap

        return kwargs

    def _run(
        self,
        url: str,
        *,
        limit: int | None = None,
        domain_filter: str | None = None,
        sitemap: str | None = None,
        locale: str | None = None,
        country: str | None = None,
    ) -> dict[str, Any]:
        """Execute map synchronously."""
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        map_kwargs = self._build_map_kwargs(
            url=url,
            limit=limit,
            domain_filter=domain_filter,
            sitemap=sitemap,
            locale=locale,
            country=country,
        )

        with handle_api_errors(operation="map"):
            response = self._sync_client.map(**map_kwargs)
            return response.model_dump()

    async def _arun(
        self,
        url: str,
        *,
        limit: int | None = None,
        domain_filter: str | None = None,
        sitemap: str | None = None,
        locale: str | None = None,
        country: str | None = None,
    ) -> dict[str, Any]:
        """Execute map asynchronously."""
        if self._async_client is None:
            msg = "Async client not initialized"
            raise RuntimeError(msg)

        map_kwargs = self._build_map_kwargs(
            url=url,
            limit=limit,
            domain_filter=domain_filter,
            sitemap=sitemap,
            locale=locale,
            country=country,
        )

        with handle_api_errors(operation="map"):
            response = await self._async_client.map(**map_kwargs)
            return response.model_dump()
