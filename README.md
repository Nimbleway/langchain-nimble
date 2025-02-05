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
`NimbleSearchRetriever` class exposes LLMs from Nimble.

```python
from langchain_nimble import NimbleSearchRetriever

retriever = NimbleSearchRetriever()
retriever.invoke("Nimbleway")
```

For the full reference with examples please see [our documentation](docs/nimbleway.ipynb).
