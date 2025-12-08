"""Test basic imports to verify package installation."""

from langchain_nimble import __all__

EXPECTED_ALL = [
    "NimbleExtractRetriever",
    "NimbleExtractTool",
    "NimbleSearchRetriever",
    "NimbleSearchTool",
    "__version__",
]


def test_all_imports() -> None:
    """Test that all expected imports are in `__all__`."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
