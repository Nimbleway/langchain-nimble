"""Nimble Search API retriever implementation."""

from typing import Any

import httpx
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langchain_core.documents.base import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.utils import from_env, secret_from_env
from pydantic import Field, SecretStr

from ._types import ParsingType, SearchEngine


class NimbleSearchRetriever(BaseRetriever):
    """Nimbleway Search API retriever.

    Allows you to retrieve search results from Google, Bing, and Yandex.
    Visit https://www.nimbleway.com/ and sign up to receive
    an API key and to see more info.

    Args:
        api_key: The API key for Nimbleway.
        base_url: Base URL for the API. Default is production endpoint.
        search_engine: The search engine to use. Default is Google.
        render: Whether to render the results web sites. Default is True.
        locale: The locale to use. Default is "en".
        country: The country to use. Default is "US".
        parsing_type: The parsing type to use. Default is "plain_text".
        links: The list of links to search for. Default is None. (if enabled will
        ignore the query)
    """

    nimble_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env("NIMBLE_API_KEY", default=""),
    )
    """API key for Nimble Search API.

    If a value isn't passed in, will attempt to read from `NIMBLE_API_KEY`
    environment variable.
    """

    nimble_api_url: str | None = Field(
        alias="base_url",
        default_factory=from_env(
            "NIMBLE_API_URL",
            default="https://nimble-retriever.webit.live",
        ),
    )
    """Base URL for API requests.

    If a value isn't passed in, will attempt to read from `NIMBLE_API_URL`
    environment variable. Defaults to `https://nimble-retriever.webit.live`.
    """

    k: int = 3
    search_engine: SearchEngine = SearchEngine.GOOGLE
    render: bool = False
    locale: str = "en"
    country: str = "US"
    parsing_type: ParsingType = ParsingType.PLAIN_TEXT
    links: list[str] = []

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun, **kwargs: Any
    ) -> list[Document]:
        request_body = {
            "query": query,
            "num_results": kwargs.get("k", self.k),
            "search_engine": kwargs.get("search_engine", self.search_engine).value,
            "render": kwargs.get("render", self.render),
            "locale": kwargs.get("locale", self.locale),
            "country": kwargs.get("country", self.country),
            "parsing_type": kwargs.get("parsing_type", self.parsing_type).value,
            "links": kwargs.get("links", self.links),
        }
        route = "extract" if self.links else "search"
        response = httpx.post(
            f"{self.nimble_api_url}/{route}",
            json=request_body,
            headers={
                "Authorization": f"Basic {self.nimble_api_key.get_secret_value()}",
                "Content-Type": "application/json",
            },
            timeout=60,
        )
        response.raise_for_status()
        raw_json_content = response.json()
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
