# langchain-nimble

This package contains the LangChain integration with Nimble

## Installation

```bash
pip install -U langchain-nimble
```

And you should configure credentials by setting the following environment variables:

* TODO: fill this out

## Chat Models

`ChatNimble` class exposes chat models from Nimble.

```python
from langchain_nimble import ChatNimble

llm = ChatNimble()
llm.invoke("Sing a ballad of LangChain.")
```

## Embeddings

`NimbleEmbeddings` class exposes embeddings from Nimble.

```python
from langchain_nimble import NimbleEmbeddings

embeddings = NimbleEmbeddings()
embeddings.embed_query("What is the meaning of life?")
```

## LLMs
`NimbleLLM` class exposes LLMs from Nimble.

```python
from langchain_nimble import NimbleLLM

llm = NimbleLLM()
llm.invoke("The meaning of life is")
```
