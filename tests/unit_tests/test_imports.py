"""Test basic imports to verify package installation."""

from langchain_nimble import __all__

EXPECTED_ALL = [
    "BrowserlessDriver",
    "NimbleAgentGetTool",
    "NimbleAgentListTool",
    "NimbleAgentRunTool",
    "NimbleCrawlTool",
    "NimbleExtractRetriever",
    "NimbleExtractTool",
    "NimbleMapTool",
    "NimbleSearchRetriever",
    "NimbleSearchTool",
    "NimbleToolkit",
    "SearchDepth",
    "__version__",
]


def test_all_imports() -> None:
    """Test that all expected imports are in `__all__`."""
    assert sorted(EXPECTED_ALL) == sorted(__all__)
