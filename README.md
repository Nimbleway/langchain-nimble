# langchain-nimble

> **Production-grade LangChain integration for Nimble's Web Search & Content Extraction API**

[![PyPI version](https://badge.fury.io/py/langchain-nimble.svg)](https://badge.fury.io/py/langchain-nimble)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

langchain-nimble provides powerful web search and content extraction capabilities for LangChain applications. Built on Nimble's production-tested API, it offers both retrievers and tools for seamless integration with LangChain agents and chains.

## Features

- ✨ **Dual Interface**: Retrievers for chains, Tools for agents
- 🔍 **Deep Search Mode**: Full page content extraction, not just snippets
- 🤖 **LLM Answers**: Optional AI-generated answer summaries
- 🎯 **Focus Modes**: Specialized search (general, news, location, shopping, geo, social)
- 🛍️ **AI-Powered WSA**: Web Search Agents for shopping, geo, and social media
- ⏰ **Time Range Filtering**: Quick recency filters (hour, day, week, month, year)
- 📅 **Date Filtering**: Search by specific date ranges
- 🌐 **Domain Control**: Include/exclude specific domains
- ⚡ **Full Async Support**: Both sync and async implementations
- 🔄 **Smart Retry Logic**: Automatic retry with exponential backoff
- 📊 **Multiple Formats**: Plain text, Markdown (default), or HTML output

## Installation

```bash
pip install -U langchain-nimble
```

## Quick Start

### 1. Get Your API Key

Sign up at [Nimbleway](https://nimbleway.com/) to get your API key.

### 2. Set Environment Variable

```bash
export NIMBLE_API_KEY="your-api-key-here"
```

Or pass it directly: `NimbleSearchRetriever(api_key="your-key")`

### 3. Basic Usage

```python
from langchain_nimble import NimbleSearchRetriever

# Create a retriever
retriever = NimbleSearchRetriever(max_results=5)

# Search (sync or async with ainvoke)
documents = retriever.invoke("latest developments in AI")

for doc in documents:
    print(f"{doc.metadata['title']}\n{doc.metadata['url']}\n")
```

## Retrievers

Retrievers return LangChain `Document` objects, ideal for RAG pipelines and chains.

### NimbleSearchRetriever

#### Basic Search

```python
from langchain_nimble import NimbleSearchRetriever

# Fast search - returns metadata only
retriever = NimbleSearchRetriever(
    max_results=5,
    deep_search=False  # Fast, metadata only
)
docs = retriever.invoke("Python best practices 2024")
```

#### Deep Search

Fetch full page content from each result:

```python
retriever = NimbleSearchRetriever(
    max_results=3,
    deep_search=True  # Full page content
)
docs = retriever.invoke("comprehensive guide to FastAPI")
```

#### Advanced Filtering

```python
# Domain filtering
retriever = NimbleSearchRetriever(
    max_results=5,
    include_domains=["python.org", "docs.python.org"],
    exclude_domains=["pinterest.com"]
)

# Date filtering
retriever = NimbleSearchRetriever(
    max_results=10,
    start_date="2024-01-01",
    end_date="2024-12-31",
    focus="news"
)

# Time range filtering
recent_retriever = NimbleSearchRetriever(
    time_range="week"  # hour, day, week, month, year
)

# Focus-based search
news_retriever = NimbleSearchRetriever(focus="news")
location_retriever = NimbleSearchRetriever(focus="location")
shopping_retriever = NimbleSearchRetriever(focus="shopping")  # AI-powered WSA
```

#### LLM Answer Generation

Get AI-generated answers (only with `deep_search=False`):

```python
retriever = NimbleSearchRetriever(
    max_results=5,
    deep_search=False,
    include_answer=True
)
docs = retriever.invoke("What is the capital of France?")

# First doc contains the LLM answer if available
if docs and docs[0].metadata.get("entity_type") == "answer":
    print(f"Answer: {docs[0].page_content}")
```

### NimbleExtractRetriever

Extract content from specific URLs:

```python
from langchain_nimble import NimbleExtractRetriever

retriever = NimbleExtractRetriever()
docs = retriever.invoke("https://www.python.org/about/")

# Advanced options
retriever = NimbleExtractRetriever(
    driver="vx8",      # Optional: vx6, vx8, vx8-pro, vx10, vx10-pro, vx12, vx12-pro
    wait=3000,         # Wait for dynamic content (ms)
    output_format="markdown"  # plain_text, markdown (default), simplified_html
)
```

## Tools for Agents

Tools provide structured input schemas for agent integration.

### NimbleSearchTool

```python
from langchain_nimble import NimbleSearchTool
from langchain.agents import create_agent

# Create agent with search tool
search_tool = NimbleSearchTool()
agent = create_agent(
    model="gpt-4o",
    tools=[search_tool]
)

# Agent searches the web
response = agent.invoke({
    "messages": [{"role": "user", "content": "What are the latest developments in quantum computing?"}]
})
```

### NimbleExtractTool

```python
from langchain_nimble import NimbleExtractTool

extract_tool = NimbleExtractTool()

# Extract single or multiple URLs
result = extract_tool.invoke({
    "urls": ["https://www.langchain.com/"]
})

# Batch extraction (up to 20 URLs)
result = extract_tool.invoke({
    "urls": [
        "https://docs.python.org/3/",
        "https://www.langchain.com/",
        "https://www.anthropic.com/"
    ],
    "driver": "vx8",
    "wait": 5000
})
```

### Multi-Tool Agent

```python
from langchain_nimble import NimbleSearchTool, NimbleExtractTool
from langchain.agents import create_agent

search_tool = NimbleSearchTool()
extract_tool = NimbleExtractTool()

agent = create_agent(
    model="gpt-4o",
    tools=[search_tool, extract_tool]
)

# Agent can search, then extract specific URLs
response = agent.invoke({
    "messages": [{"role": "user", "content": "Find recent LangChain articles and summarize the top one"}]
})
```

## Parameter Reference

### Search Parameters (NimbleSearchRetriever & NimbleSearchTool)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | `None` | API key (or set `NIMBLE_API_KEY`) |
| `max_results` | `int` | `3` / `10`* | Number of results (1-100). Alias: `num_results` |
| `focus` | `str` | `"general"` | Search focus mode |
| `deep_search` | `bool` | `True` / `False`* | Full content vs. metadata only |
| `include_answer` | `bool` | `False` | LLM answer (requires `deep_search=False`) |
| `time_range` | `str` | `None` | Recency filter - hour, day, week, month, year |
| `include_domains` | `list[str]` | `None` | Domain whitelist |
| `exclude_domains` | `list[str]` | `None` | Domain blacklist |
| `start_date` | `str` | `None` | Filter after date (YYYY-MM-DD or YYYY) |
| `end_date` | `str` | `None` | Filter before date (YYYY-MM-DD or YYYY) |
| `locale` | `str` | `"en"` | Language/locale (e.g., `fr`, `es`) |
| `country` | `str` | `"US"` | Country code (e.g., `UK`, `FR`) |
| `output_format` | `str` | `"markdown"` | Content format - plain_text, markdown, simplified_html |

\* Defaults differ: Retriever uses `max_results=3, deep_search=True`; Tool uses `max_results=10, deep_search=False`

### Extract Parameters (NimbleExtractRetriever & NimbleExtractTool)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | `None` | API key (or set `NIMBLE_API_KEY`) |
| `driver` | `str \| None` | `None` | Optional driver: vx6, vx8, vx8-pro, vx10, vx10-pro, vx12, vx12-pro. API auto-selects if not specified. |
| `wait` | `int \| None` | `None` | Wait before extraction (milliseconds) |
| `locale` | `str` | `"en"` | Language/locale |
| `country` | `str` | `"US"` | Country code |
| `output_format` | `str` | `"markdown"` | Content format - plain_text, markdown, simplified_html |

## Response Formats

### Document Structure (Retrievers)

```python
Document(
    page_content="Full content...",
    metadata={
        "title": "Page Title",
        "url": "https://example.com",
        "description": "Page description...",
        "position": 1,
        "entity_type": "organic"  # or "answer"
    }
)
```

### Tool Response (JSON)

```python
{
    "results": [
        {
            "title": "Title",
            "url": "https://...",
            "description": "...",
            "content": "Full content...",
            "metadata": {
                "position": 1,
                "entity_type": "organic"
            }
        }
    ]
}
```

## Best Practices

### Deep Search vs. Regular Search

**Use `deep_search=True` for:**
- RAG applications needing full context
- Content analysis and summarization
- In-depth research tasks

**Use `deep_search=False` for:**
- Quick lookups (5-10x faster)
- Getting lists of URLs
- When you'll extract specific URLs later

### Tools vs. Retrievers

**Retrievers**: Use in chains, RAG pipelines, vector store integration
**Tools**: Use with agents that need dynamic search control

### Filtering Tips

- **Academic research**: `include_domains=["edu", "scholar.google.com"]`
- **Documentation**: `include_domains=["docs.python.org", "readthedocs.io"]`
- **Remove noise**: `exclude_domains=["pinterest.com", "facebook.com"]`
- **Recent news**: `start_date="2024-01-01", focus="news"`
- **Historical**: `start_date="2020", end_date="2021"`

### Error Handling

Automatic retry with exponential backoff for 5xx errors. For custom handling:

```python
import httpx
from langchain_nimble import NimbleSearchRetriever

retriever = NimbleSearchRetriever()

try:
    docs = retriever.invoke("query")
except httpx.HTTPStatusError as e:
    print(f"HTTP {e.response.status_code}")
except httpx.RequestError as e:
    print(f"Network error: {e}")
```

### Performance Tips

1. Use async (`ainvoke`) for concurrent requests
2. Batch URLs with `NimbleExtractTool` (up to 20)
3. Request only needed results (`max_results`)
4. Let API auto-select driver, or use lower driver levels (vx6/vx8) unless advanced rendering needed
5. Avoid `wait` parameter for static content

## Examples & Documentation

- **Examples**: [examples/](https://github.com/Nimbleway/langchain-nimble/tree/main/examples)
- **API Docs**: [docs.nimbleway.com](https://docs.nimbleway.com/)
- **LangChain**: [python.langchain.com](https://python.langchain.com/)

## Contributing

Contributions welcome! Please submit Pull Requests.

1. Fork the repository
2. Create feature branch (`git checkout -b feature/name`)
3. Commit changes (`git commit -m 'Add feature'`)
4. Push branch (`git push origin feature/name`)
5. Open Pull Request

## Support

- **Issues**: [GitHub Issues](https://github.com/Nimbleway/langchain-nimble/issues)
- **Docs**: [docs.nimbleway.com](https://docs.nimbleway.com/)
- **Website**: [nimbleway.com](https://nimbleway.com/)

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

Built with ❤️ by the Nimbleway team
