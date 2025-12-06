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
You can get your API key from [Nimble's website](https://nimbleway.com/)
Just, go to the log-in page and sign up for a new account. After that, you can get your API key from the dashboard.

## Retrievers

### Search Retriever
`NimbleSearchRetriever` retrieves search results from Google, Bing, and Yandex.

```python
from langchain_nimble import NimbleSearchRetriever

retriever = NimbleSearchRetriever(num_results=5, deep_search=True)
await retriever.ainvoke("Nimbleway")
```

### Extract Retriever
`NimbleExtractRetriever` extracts content from specific URLs.

```python
from langchain_nimble import NimbleExtractRetriever

retriever = NimbleExtractRetriever(
    links=[
        "https://example.com/page1",
        "https://example.com/page2"
    ]
)
await retriever.ainvoke("query")
```

For the full reference with examples please see [our documentation](https://github.com/Nimbleway/langchain-nimble/blob/main/docs/nimbleway.ipynb).
