"""LangChain integration for Nimble's web search and content retrieval API."""

from importlib import metadata

from langchain_nimble.retrievers import NimbleExtractRetriever, NimbleSearchRetriever
from langchain_nimble.search_tool import NimbleSearchTool

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "NimbleExtractRetriever",
    "NimbleSearchRetriever",
    "NimbleSearchTool",
    "__version__",
]
