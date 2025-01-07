from importlib import metadata

from langchain_nimble.chat_models import ChatNimble
from langchain_nimble.document_loaders import NimbleLoader
from langchain_nimble.embeddings import NimbleEmbeddings
from langchain_nimble.retrievers import NimbleRetriever
from langchain_nimble.toolkits import NimbleToolkit
from langchain_nimble.tools import NimbleTool
from langchain_nimble.vectorstores import NimbleVectorStore

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    # Case where package metadata is not available.
    __version__ = ""
del metadata  # optional, avoids polluting the results of dir(__package__)

__all__ = [
    "ChatNimble",
    "NimbleVectorStore",
    "NimbleEmbeddings",
    "NimbleLoader",
    "NimbleRetriever",
    "NimbleToolkit",
    "NimbleTool",
    "__version__",
]
