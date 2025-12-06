"""Integration tests for NimbleSearchRetriever.

These tests make real API calls and require NIMBLE_API_KEY environment variable.
Run with: make integration_tests
"""

import os

import pytest
from langchain_core.documents.base import Document

from langchain_nimble import NimbleSearchRetriever
from langchain_nimble.retrievers import ParsingType


@pytest.fixture
def api_key() -> str:
    """Get API key from environment."""
    key = os.environ.get("NIMBLE_API_KEY")
    if not key:
        pytest.skip("NIMBLE_API_KEY environment variable not set")
    return key


def test_retriever_search_basic(api_key: str) -> None:
    """Test basic search functionality with real API."""
    retriever = NimbleSearchRetriever(
        api_key=api_key, k=3, parsing_type=ParsingType.MARKDOWN
    )
    documents = retriever.invoke("LangChain framework")

    assert len(documents) > 0
    assert len(documents) <= 3

    doc = documents[0]
    assert isinstance(doc, Document)
    assert doc.page_content
    assert "title" in doc.metadata
    assert "url" in doc.metadata
    assert doc.metadata["url"].startswith("http")


def test_retriever_extract_from_links(api_key: str) -> None:
    """Test extraction from specific URLs."""
    retriever = NimbleSearchRetriever(
        api_key=api_key, links=["https://www.langchain.com/"], render=True
    )
    documents = retriever.invoke("ignored query")

    assert len(documents) > 0
    assert documents[0].page_content
    assert documents[0].metadata["url"]


def test_retriever_invalid_api_key() -> None:
    """Test retriever handles invalid API key gracefully."""
    retriever = NimbleSearchRetriever(api_key="invalid_key", k=1)

    with pytest.raises(Exception):
        retriever.invoke("test query")
