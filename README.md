# langchain-nimble

> **Production-grade LangChain integration for Nimble's Web Search & Content Extraction API**

[![PyPI version](https://badge.fury.io/py/langchain-nimble.svg)](https://badge.fury.io/py/langchain-nimble)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

langchain-nimble provides powerful web search and content extraction capabilities for LangChain applications. Built on the official [Nimble Python SDK](https://pypi.org/project/nimble_python/), it offers both retrievers and tools for seamless integration with LangChain agents and chains.

## Features

- ✨ **Dual Interface**: Retrievers for chains, Tools for agents
- 🔍 **Search Depth Levels**: lite (metadata), fast (Enterprise), deep (full content)
- 🤖 **LLM Answers**: Optional AI-generated answer summaries
- 🎯 **Focus Modes**: Specialized search (general, news, location, shopping, geo, social)
- 📋 **Extract Templates**: Structured site scraping (`nimble_extract_template_*`)
- 🤖 **Agent API V2**: Resumable Web Search Agent research (`start` / `status` / `result`)
- ⏰ **Time Range Filtering**: Quick recency filters (hour, day, week, month, year)
- 📅 **Date Filtering**: Search by specific date ranges
- 🌐 **Domain Control**: Include/exclude specific domains
- ⚡ **Full Async Support**: Both sync and async implementations
- 🔄 **Smart Retry Logic**: Built-in retry via Nimble SDK
- 📊 **Markdown Output**: Clean markdown content from any page

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

# Lite search - returns metadata only (default)
retriever = NimbleSearchRetriever(
    max_results=5,
    search_depth="lite"
)
docs = retriever.invoke("Python best practices 2024")
```

#### Deep Search

Fetch full page content from each result:

```python
retriever = NimbleSearchRetriever(
    max_results=3,
    search_depth="deep"  # Full page content extraction
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

Get AI-generated answers:

```python
retriever = NimbleSearchRetriever(
    max_results=5,
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

# With render wait for dynamic content
retriever = NimbleExtractRetriever(
    driver="vx8",      # Optional: vx6, vx8, vx8-pro, vx10, vx10-pro, vx12, vx12-pro
    wait=3000,         # Wait for dynamic content (ms) - uses browser_actions
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
    model="claude-haiku-4-5",
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

# Extract a URL - returns markdown string
result = extract_tool.invoke({
    "url": "https://www.langchain.com/"
})
```

### Extract Templates (structured site scraping)

Use when you need a named template (e.g. product pages) with structured params — distinct from URL markdown extract and from Agent API V2 research.

```python
from langchain_nimble import NimbleToolkit

toolkit = NimbleToolkit(include_extract_templates=True)
tools = toolkit.get_tools()
# nimble_extract_template_list → get → run
```

Or import tools directly: `NimbleExtractTemplateListTool`, `NimbleExtractTemplateGetTool`, `NimbleExtractTemplateRunTool`.

> **Deprecated:** `NimbleAgentListTool` / `Get` / `Run` (`nimble_agent_*`) are aliases that now wrap Extract Templates. Prefer the `nimble_extract_template_*` names. They do **not** call Agent API V2.

### Agent API V2 (Web Search Agents / research)

Resumable research agents under `/v2/agents/*`. **Distinct from Extract Templates.**

LangChain tool sessions are typically **stateless**, so prefer **Mode 1**
(`agent_name` create-or-reuse) on `nimble_web_search_agent_run_start`. Use
**Mode 2** (`agent_id` / `wsa_…`) when your app persists the id. Omit both for
**Mode 3** anonymous one-shot (response still includes `web_search_agent_id`).

Start / status / result are **separate tools** — do not hide multi-minute
polling in one call. Runs often take **3–15 minutes**.

```python
from langchain_nimble import NimbleToolkit

toolkit = NimbleToolkit(include_web_search_agents=True)
tools = toolkit.get_tools()
# nimble_web_search_agents_list, nimble_web_search_agent_templates_list,
# nimble_web_search_agent_create, nimble_web_search_agent_run_start,
# nimble_web_search_agent_run_status, nimble_web_search_agent_run_result
```

#### Modes (runnable sketches)

**Mode 1 — create-or-reuse by name (default for stateless hosts):**

```python
from langchain_nimble import NimbleAgentRunStartTool, NimbleAgentRunStatusTool, NimbleAgentRunResultTool

start = NimbleAgentRunStartTool()
started = start.invoke({
    "agent_name": "integrations_research_bot",
    "use_case": "research",
    "effort": "medium",
    "skill": "Focus on official docs and changelogs",
    "input": "Summarize recent Agent API v2 changes for integrators.",
})
# started["id"] -> task_run_… ; started["web_search_agent_id"] -> wsa_…
# Reuse the same agent_name later; a failed first run does not brick the name.
```

**Mode 2 — persist `wsa_…`:**

```python
from langchain_nimble import NimbleAgentCreateTool, NimbleAgentRunStartTool

create = NimbleAgentCreateTool()
agent = create.invoke({
    "agent_name": "enrich_bot",
    "use_case": "enrichment",
    "skill": "Company firmographics",
    "output_schema": {"type": "object", "properties": {"domain": {"type": "string"}}},
})
start = NimbleAgentRunStartTool()
started = start.invoke({
    "agent_id": agent["id"],
    "input": "Enrich this company row",
    "input_data": [{"domain": "example.com"}],
    "effort": "medium",
})
```

**Mode 3 — anonymous one-shot:** omit `agent_id` and `agent_name`; still read
`web_search_agent_id` from the start response if you want to graduate to Mode 2.

#### Effort tiers

| Tier | Guidance |
|------|----------|
| `low` | ~15–17s; may skip live research (0 sources / low confidence) |
| `medium` | ~90–160s observed — good default for real research |
| `high` / `x-high` / `max` | Longer; design UX for **3–15 minutes** wall time |

#### `use_case` (locked, not a silent per-run override)

| Value | Output | When |
|-------|--------|------|
| `research` | `output.type: "text"` + citations | Free-form cited answer |
| `enrichment` | `output.type: "json"` | Fill `input_data` against a schema |
| `dataset_building` | `output.type: "json"` | Structured table from scratch (API requires `effort` `high`+) |

Set **once** when the agent is created (Mode 1 first call, Mode 3, or
`nimble_web_search_agent_create`). Against an existing agent: omit or pass the
**same** value — a different value returns **422**.

#### Overrides vs persist

On an **existing** agent, run-level `sources` / `output_schema` / `skill` are
**one-time** (they do not mutate the stored agent). On **Mode 1 first create**,
those fields **are** stored. `input_data` is always run-only (enrichment
payload ≠ schema). `use_case` is never a silent override.

#### `sources` fields

```python
sources = {
    "allow": [{"title": "Official filings", "domains": ["sec.gov"], "order": 0}],
    "block": [{"title": "Junk", "domains": ["example.com"], "order": 0}],
    "avoid": "free-text domains or source types to avoid",
    "prioritize": "free-text domains or source types to prefer",
}
```

#### Events (intentional gap)

API supports `enable_events: true` + `GET …/runs/{run_id}/events` (SSE). This
package does **not** expose a separate events tool yet — use start/status/result.

Attribution: every request sends `X-Client-Source: langchain-nimble`.

### Multi-Tool Agent

```python
from langchain_nimble import NimbleSearchTool, NimbleExtractTool
from langchain.agents import create_agent

search_tool = NimbleSearchTool()
extract_tool = NimbleExtractTool()

agent = create_agent(
    model="claude-haiku-4-5",
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
| `search_depth` | `str` | `"lite"` | Search depth: lite, fast (Enterprise), deep |
| `include_answer` | `bool` | `False` | LLM answer summary |
| `time_range` | `str` | `None` | Recency filter - hour, day, week, month, year |
| `include_domains` | `list[str]` | `None` | Domain whitelist |
| `exclude_domains` | `list[str]` | `None` | Domain blacklist |
| `start_date` | `str` | `None` | Filter after date (YYYY-MM-DD or YYYY) |
| `end_date` | `str` | `None` | Filter before date (YYYY-MM-DD or YYYY) |
| `locale` | `str` | `"en"` | Language/locale (e.g., `fr`, `es`) |
| `country` | `str` | `"US"` | Country code (e.g., `UK`, `FR`) |

\* Defaults differ: Retriever uses `max_results=3, search_depth="lite"`; Tool uses `max_results=10, search_depth="lite"`

### Extract Parameters (NimbleExtractRetriever)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `api_key` | `str \| None` | `None` | API key (or set `NIMBLE_API_KEY`) |
| `driver` | `str \| None` | `None` | Browser driver: vx6, vx8, vx8-pro, vx10, vx10-pro, vx12, vx12-pro |
| `wait` | `int \| None` | `None` | Render wait in milliseconds (uses browser_actions) |
| `locale` | `str` | `"en"` | Language/locale |
| `country` | `str` | `"US"` | Country code |

### NimbleExtractTool

The extract tool accepts a single `url` parameter and returns the page content as a markdown string.

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

### Search Tool Response (JSON)

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

### Extract Tool Response (String)

The extract tool returns a markdown string directly.

## Best Practices

### Search Depth Levels

**Use `search_depth="deep"` for:**
- RAG applications needing full context
- Content analysis and summarization
- In-depth research tasks

**Use `search_depth="lite"` (default) for:**
- Quick lookups
- Getting lists of URLs
- When you'll extract specific URLs later

**Use `search_depth="fast"` for (Enterprise only):**
- Production workloads needing rich content at low latency

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

The SDK handles retries automatically. For custom error handling:

```python
from langchain_nimble import NimbleSearchRetriever

retriever = NimbleSearchRetriever()

try:
    docs = retriever.invoke("query")
except ValueError as e:
    print(f"API error: {e}")
```

### Performance Tips

1. Use async (`ainvoke`) for concurrent requests
2. Request only needed results (`max_results`)
3. Let API auto-select driver, or use lower driver levels (vx6/vx8) unless advanced rendering needed
4. Avoid `wait` parameter for static content

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
