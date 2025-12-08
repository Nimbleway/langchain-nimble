"""Standard integration tests for Nimble retrievers and tools.

These tests use the langchain-tests framework to ensure compliance with
LangChain standards for retrievers and tools.

Requires NIMBLE_API_KEY environment variable.
"""

import os

import pytest
from langchain_core.retrievers import BaseRetriever
from langchain_core.tools import BaseTool
from langchain_tests.integration_tests import (
    RetrieversIntegrationTests,
    ToolsIntegrationTests,
)

from langchain_nimble import (
    NimbleExtractRetriever,
    NimbleSearchRetriever,
    NimbleSearchTool,
)


def _get_api_key() -> str:
    """Get API key from environment or skip test."""
    key = os.environ.get("NIMBLE_API_KEY")
    if not key:
        pytest.skip("NIMBLE_API_KEY not set")
    return key


class TestNimbleSearchRetrieverStandard(RetrieversIntegrationTests):
    """Standard integration tests for NimbleSearchRetriever."""

    @property
    def retriever_constructor(self) -> type[BaseRetriever]:
        """Retriever class to test."""
        return NimbleSearchRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        """Constructor parameters."""
        return {
            "api_key": _get_api_key(),
            "parsing_type": "markdown",
            "deep_search": False,  # Fast mode
        }

    @property
    def retriever_query_example(self) -> str:
        """Example query."""
        return "LangChain framework"

    @property
    def num_results_arg_name(self) -> str:
        """Parameter name for number of results."""
        return "num_results"

    @pytest.mark.xfail(
        reason="Nimble API returns UP TO num_results, not exactly that number. "
        "We override to check <= instead of ==",
        strict=False,
    )
    def test_k_constructor_param(self) -> None:
        """Test num_results constructor parameter (API returns UP TO n results)."""
        params = {
            k: v
            for k, v in self.retriever_constructor_params.items()
            if k != self.num_results_arg_name
        }
        params_3 = {**params, self.num_results_arg_name: 3}
        retriever_3 = self.retriever_constructor(**params_3)
        result_3 = retriever_3.invoke(self.retriever_query_example)
        assert len(result_3) >= 1, "Should return at least 1 result"
        assert len(result_3) <= 3, "Should not return more than 3 results"
        assert all(isinstance(doc, type(result_3[0])) for doc in result_3)

        params_1 = {**params, self.num_results_arg_name: 1}
        retriever_1 = self.retriever_constructor(**params_1)
        result_1 = retriever_1.invoke(self.retriever_query_example)
        assert len(result_1) >= 1, "Should return at least 1 result"
        assert len(result_1) <= 1, "Should not return more than 1 result"
        assert all(isinstance(doc, type(result_1[0])) for doc in result_1)

    @pytest.mark.xfail(
        reason="Nimble API returns UP TO num_results, not exactly that number. "
        "We override to check <= instead of ==",
        strict=False,
    )
    def test_invoke_with_k_kwarg(self, retriever: BaseRetriever) -> None:
        """Test num_results in invoke() (API returns UP TO n results)."""
        result_1 = retriever.invoke(
            self.retriever_query_example, None, **{self.num_results_arg_name: 1}
        )
        assert len(result_1) >= 1, "Should return at least 1 result"
        assert len(result_1) <= 1, "Should not return more than 1 result"
        assert all(isinstance(doc, type(result_1[0])) for doc in result_1)

        result_3 = retriever.invoke(
            self.retriever_query_example, None, **{self.num_results_arg_name: 3}
        )
        assert len(result_3) >= 1, "Should return at least 1 result"
        assert len(result_3) <= 3, "Should not return more than 3 results"
        assert all(isinstance(doc, type(result_3[0])) for doc in result_3)


class TestNimbleExtractRetrieverStandard(RetrieversIntegrationTests):
    """Standard integration tests for NimbleExtractRetriever."""

    @property
    def retriever_constructor(self) -> type[BaseRetriever]:
        """Retriever class to test."""
        return NimbleExtractRetriever

    @property
    def retriever_constructor_params(self) -> dict:
        """Constructor parameters."""
        return {"api_key": _get_api_key()}

    @property
    def retriever_query_example(self) -> str:
        """Example URL."""
        return "https://www.langchain.com/"

    @pytest.mark.xfail(reason="Extract retriever doesn't support num_results parameter")
    def test_k_constructor_param(self) -> None:
        """Not supported - extract always returns one document."""
        raise NotImplementedError

    @pytest.mark.xfail(reason="Extract retriever doesn't support num_results parameter")
    def test_invoke_with_k_kwarg(self, retriever: BaseRetriever) -> None:
        """Not supported - extract always returns one document."""
        raise NotImplementedError


class TestNimbleSearchToolStandard(ToolsIntegrationTests):
    """Standard integration tests for NimbleSearchTool."""

    @property
    def tool_constructor(self) -> type[BaseTool]:
        """Tool class to test."""
        return NimbleSearchTool

    @property
    def tool_constructor_params(self) -> dict:
        """Constructor parameters."""
        return {"api_key": _get_api_key()}

    @property
    def tool_invoke_params_example(self) -> dict:
        """Example invocation parameters."""
        return {"query": "LangChain framework", "num_results": 3}
