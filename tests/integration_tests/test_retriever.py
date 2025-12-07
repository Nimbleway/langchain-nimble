"""Integration tests for Nimble retrievers (async mode only).

Requires NIMBLE_API_KEY environment variable.
"""

import os

import pytest
from langchain_core.documents.base import Document

from langchain_nimble import NimbleExtractRetriever, NimbleSearchRetriever


@pytest.fixture
def api_key() -> str:
    """Get API key from environment or skip test."""
    key = os.environ.get("NIMBLE_API_KEY")
    if not key:
        pytest.skip("NIMBLE_API_KEY not set")
    return key


async def test_search_retriever_async_basic(api_key: str) -> None:
    """Test async search."""
    retriever = NimbleSearchRetriever(
        api_key=api_key, num_results=3, parsing_type="markdown"
    )
    documents = await retriever.ainvoke("LangChain framework")

    assert len(documents) > 0
    assert len(documents) <= 3

    doc = documents[0]
    assert isinstance(doc, Document)
    assert doc.page_content
    assert "title" in doc.metadata
    assert "url" in doc.metadata
    assert doc.metadata["url"].startswith("http")


async def test_extract_retriever_async_basic(api_key: str) -> None:
    """Test async extraction."""
    retriever = NimbleExtractRetriever(api_key=api_key)
    documents = await retriever.ainvoke("https://www.langchain.com/")

    assert len(documents) > 0
    assert documents[0].page_content
    assert documents[0].metadata["url"]


async def test_search_retriever_async_invalid_api_key() -> None:
    """Test async error handling for invalid API key."""
    retriever = NimbleSearchRetriever(api_key="invalid_key", num_results=1)

    with pytest.raises(Exception):
        await retriever.ainvoke("test query")


async def test_extract_retriever_async_invalid_api_key() -> None:
    """Test extract error handling for invalid API key."""
    retriever = NimbleExtractRetriever(api_key="invalid_key")

    with pytest.raises(Exception):
        await retriever.ainvoke("https://example.com")
