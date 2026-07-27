# Examples

This directory contains example scripts demonstrating how to use langchain-nimble.

## Setup

### 1. Install langchain-nimble

```bash
pip install langchain-nimble
```

### 2. Install langchain (for agent examples)

The agent examples require the `langchain` package:

```bash
pip install langchain
```

Or if using this repo:
```bash
# Install in editable mode with langchain
pip install -e . langchain
```

### 3. Set your API keys

Create a `.env` file in the project root:

```bash
NIMBLE_API_KEY=your-nimble-api-key
ANTHROPIC_API_KEY=your-anthropic-api-key
```

Or export environment variables:

```bash
export NIMBLE_API_KEY="your-nimble-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

## Examples

### Multi-Tool Web Agent (`web_search_agent.py`)

An async example with Search, Extract, Map, Crawl, and **Extract Templates**.

**Run:**
```bash
python examples/web_search_agent.py
```

**Features:**
- Claude Haiku 4.5
- Async multi-tool agent
- Extract Templates workflow (`list` → `get` → `run`) for structured site scraping

### Agent API V2 Research (`agent_api_v2.py`)

Resumable Web Search Agent lifecycle: discover/create → `run_start` → `run_status` → `run_result`.

**Run:**
```bash
python examples/agent_api_v2.py
```

This is distinct from Extract Templates — use it for multi-minute research agents, not one-shot structured scrapes.
