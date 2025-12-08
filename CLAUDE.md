# langchain-nimble - Claude Code Guidelines

Project-specific context and patterns for working on langchain-nimble.

---

## Tech Stack

- **Python:** 3.10+ (using modern union syntax: `str | None`)
- **HTTP Client:** httpx (sync + async)
- **Framework:** LangChain Core (retrievers, tools, documents)
- **Validation:** Pydantic v2 models
- **Testing:** pytest, pytest-asyncio, pytest-mock, syrupy, freezegun, langchain-tests
- **Linting:** ruff, mypy --strict

---

## Project Structure

```
langchain_nimble/
├── retrievers.py          # NimbleSearchRetriever - main retriever class
├── tools.py               # NimbleSearch, NimbleExtract - LangChain tools for agents
├── _utilities.py          # Client creation, API key handling (private)
├── _types.py              # Shared Pydantic models and type definitions (private)
└── __init__.py            # Public exports

tests/
├── unit_tests/            # Fast tests with mocks
├── integration_tests/     # Real API tests (requires NIMBLE_API_KEY)
└── conftest.py            # Shared fixtures
```

---

## Key Commands

```bash
# Sync all development dependencies
uv sync --all-groups
# Or specify individual groups:
# uv sync --group test --group lint --group typing --group dev

# Sync just core dependencies
uv sync

# Run tests (unit only, fast)
uv run pytest tests/unit_tests/

# Run integration tests (requires NIMBLE_API_KEY env var)
uv run pytest tests/integration_tests/

# Run all tests
uv run pytest

# Type checking
uv run mypy langchain_nimble/

# Linting
uv run ruff check .
uv run ruff format .
```

---

## Architecture Patterns

### Async-First Design
- All HTTP operations support both sync and async
- Use `httpx.Client` for sync, `httpx.AsyncClient` for async
- Implement both `_get_relevant_documents()` and `_aget_relevant_documents()`

### Client Initialization
- Clients initialized once in `@model_validator(mode="after")`
- Reuse clients across requests (connection pooling)
- Use utilities from `_utilities.py`: `create_sync_client()`, `create_async_client()`

### API Key Handling
- Support both parameter (`api_key="..."`) and environment variable (`NIMBLE_API_KEY`)
- Use `SecretStr` from Pydantic for API keys
- Utility: `get_api_key()` in `_utilities.py`

### Error Handling
- Retry on 5xx errors only (use `tenacity`)
- Never retry 4xx errors (client issues)
- Exponential backoff: 2s, 4s, 8s

---

## Testing Approach

### 3-Layer Testing Strategy

1. **Unit Tests** (`tests/unit_tests/`)
   - Mock HTTP calls with `unittest.mock`
   - Fast, no network access
   - Test individual functions
   - Benchmark tests for performance tracking

2. **Integration Tests** (`tests/integration_tests/`)
   - Real Nimble API calls
   - Requires `NIMBLE_API_KEY` environment variable
   - No mocks - test actual behavior

3. **LangChain Standard Tests** (`tests/integration_tests/test_standard.py`)
   - Inherit from `RetrieversIntegrationTests` (from `langchain_tests.integration_tests`)
   - Ensures LangChain compatibility and compliance
   - Tests standard retriever behavior: sync/async invoke, k parameter, Document returns
   - Use `pytest.mark.xfail` to skip tests for unsupported features

---

## Code Style Principles

### General Guidelines
- **DRY:** Extract common logic into utilities
- **Early returns:** Exit functions early with guard clauses, avoid deep nesting
- **Type hints:** All function parameters and return values must have type hints
- **Python 3.10+ syntax:** Use `str | None` not `Optional[str]`
- **Line length:** 120 characters max
- **Docstrings:** Google-style for all public APIs

### Example - Good Style
```python
def get_api_key(api_key: str | None = None) -> str:
    """Get API key from parameter or environment variable."""
    if api_key:
        return api_key

    env_key = os.environ.get("NIMBLE_API_KEY")
    if not env_key:
        raise ValueError("API key required. Set NIMBLE_API_KEY or pass api_key parameter.")

    return env_key
```

### Pydantic Models
- Use `Field()` with descriptions for all fields
- For tools: Write detailed, multi-paragraph descriptions (agents use these to decide when to use tools)
- Use validators for custom validation logic

### Example - Tool Field Description
```python
deep_search: bool = Field(
    default=False,
    description="""Enable deep search for comprehensive research.

    When enabled:
    - Fetches full page content, not just snippets
    - Takes longer but returns richer data

    Use for: In-depth research, analysis
    Don't use for: Quick facts, simple lookups
    """
)
```

### Import Organization
```python
# Standard library
import os
from typing import Literal

# Third-party
import httpx
from langchain_core.documents import Document
from pydantic import BaseModel, Field, SecretStr

# Local
from ._utilities import create_async_client, get_api_key
```

### Docstring Requirements
- **Style:** Google-style convention (enforced by pydocstyle)
- **Module-level:** All Python modules must have docstrings
- **Classes:** Multi-line with blank line after summary
- **Enums:** One-line with period
- **Functions:** Document parameters, return values, and raises

### Error Handling
- Use specific exception types, not generic `Exception`
- Include context in error messages
- Retry on 5xx errors only, never on 4xx (see Architecture Patterns section)

