"""LangChain extract tool for Nimble Content Extraction API."""

from typing import Any

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from ._utilities import _NimbleClientMixin, handle_api_errors


class NimbleExtractToolInput(BaseModel):
    """Input schema for NimbleExtractTool."""

    url: str = Field(
        description="""The URL to extract markdown content from.

        Accepts any publicly accessible URL. The tool fetches the page,
        renders JavaScript if needed, and returns the content as markdown.

        Use after nimble_search to get full content from a result URL.
        """,
    )


class NimbleExtractTool(_NimbleClientMixin, BaseTool):
    """Extract page content as markdown from a URL.

    Returns the page content as a markdown string.

    Args:
        api_key: API key for Nimbleway (or set NIMBLE_API_KEY env var).
        base_url: Override base URL for the Nimble API.
        max_retries: Maximum retry attempts for 5xx errors (default: 2).
        locale: Locale for results (default: en).
        country: Country code (default: US).
    """

    name: str = "nimble_extract"
    description: str = (
        "Extract page content as markdown from a URL. "
        "Use after search to get full content from a specific page."
    )
    args_schema: type[BaseModel] = NimbleExtractToolInput

    def _build_extract_kwargs(self, url: str) -> dict[str, Any]:
        """Build keyword arguments for SDK extract() call."""
        return {
            "url": url,
            "locale": self.locale,
            "country": self.country,
            "formats": ["markdown"],
        }

    def _run(self, url: str) -> str:
        """Execute extraction synchronously."""
        if self._sync_client is None:
            msg = "Sync client not initialized"
            raise RuntimeError(msg)

        with handle_api_errors(operation="extract"):
            response = self._sync_client.extract(**self._build_extract_kwargs(url))
            if response.data and response.data.markdown:
                return response.data.markdown
            return ""

    async def _arun(self, url: str) -> str:
        """Execute extraction asynchronously."""
        if self._async_client is None:
            msg = "Async client not initialized"
            raise RuntimeError(msg)

        with handle_api_errors(operation="extract"):
            response = await self._async_client.extract(
                **self._build_extract_kwargs(url)
            )
            if response.data and response.data.markdown:
                return response.data.markdown
            return ""
