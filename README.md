# langchain-nimble

This package contains the LangChain integration with Nimble

## Installation

```bash
pip install -U langchain-nimble
```

And you should configure credentials by setting the following environment variables:

```bash
export NIMBLE_API_KEY=<PLACEHOLDER_FOR_YOUR_NIMBLE_API_KEY>
```
You can get your API key from [Nimble's website](https://nimbleway.com/).
Just go to the log-in page and sign up for a new account. After that, you can get your API key from the dashboard.

## Retrievers

### Search Retriever
`NimbleSearchRetriever` retrieves search results with full page content extraction. Supports general, news, and location topics.

```python
from langchain_nimble import NimbleSearchRetriever

retriever = NimbleSearchRetriever(num_results=3, topic="general")

# Async usage:
# import asyncio
# async def main():
#     documents = await retriever.ainvoke("latest LangChain release")
# asyncio.run(main())

# Synchronous usage:
documents = retriever.invoke("latest LangChain release")
```

### Extract Retriever
`NimbleExtractRetriever` extracts clean content from a single URL and returns one document.

```python
from langchain_nimble import NimbleExtractRetriever

retriever = NimbleExtractRetriever()

# Async usage:
# import asyncio
# async def main():
#     document = await retriever.ainvoke("https://www.langchain.com")
# asyncio.run(main())

# Synchronous usage:
document = retriever.invoke("https://www.langchain.com")
```

For the full reference with examples please see [our documentation](https://github.com/Nimbleway/langchain-nimble/blob/main/docs/nimbleway.ipynb).
