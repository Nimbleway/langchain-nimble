"""Standard unit tests for NimbleSearchRetriever."""

import pytest
from pytest_benchmark.fixture import BenchmarkFixture  # type: ignore[import-untyped]

from langchain_nimble import NimbleSearchRetriever


@pytest.mark.benchmark
def test_nimble_retriever_init_time(benchmark: BenchmarkFixture) -> None:
    """Test NimbleSearchRetriever initialization time."""

    def _init_nimble_retriever() -> None:
        for _ in range(10):
            NimbleSearchRetriever(api_key="test_key")

    benchmark(_init_nimble_retriever)
