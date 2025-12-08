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

An async example showing how to create an agent with both search and content extraction capabilities.

**Run:**
```bash
python examples/web_search_agent.py
```

**Features:**
- ✅ Uses Claude Haiku 4.5 (fast and cost-effective)
- ✅ Async implementation for better performance
- ✅ Multi-tool agent (NimbleSearchTool + NimbleExtractTool)
- ✅ Custom system prompt for optimal behavior
- ✅ Demonstrates search and extract workflows
- ✅ Shows how to extract and display agent responses

