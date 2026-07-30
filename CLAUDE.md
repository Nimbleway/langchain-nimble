# langchain-nimble - Claude Code Guidelines

Project-specific context and patterns for working on langchain-nimble.

---

## Tech Stack

- **Python:** 3.10+ (using modern union syntax: `str | None`)
- **HTTP Client:** nimble_python SDK (wraps httpx internally)
- **Framework:** LangChain Core (retrievers, tools, documents)
- **Validation:** Pydantic v2 models
- **Testing:** pytest, pytest-asyncio, pytest-mock, syrupy, freezegun, langchain-tests
- **Linting:** ruff, mypy --strict

---

## Project Structure

```
langchain_nimble/
├── retrievers.py          # NimbleSearchRetriever, NimbleExtractRetriever
├── toolkit.py             # NimbleToolkit (BaseToolkit) - groups all tools
├── tools/                 # LangChain tools package
│   ├── search_tool.py     # NimbleSearchTool
│   ├── extract_tool.py    # NimbleExtractTool (URL → markdown via extract.run)
│   ├── extract_template_tool.py  # Extract Templates list/get/run
│   ├── map_tool.py        # NimbleMapTool
│   ├── crawl_tool.py      # NimbleCrawlTool (async with polling)
│   ├── agents_v2_tool.py  # Agent API V2 list/create/start/status/result
│   └── agent_tool.py      # Deprecated NimbleAgent* aliases → Extract Templates
├── _utilities.py          # _NimbleClientMixin, handle_api_errors (private)
├── _types.py              # Shared enums: SearchDepth, SearchFocus, etc. (private)
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
- Use `nimble_python.Nimble` for sync, `nimble_python.AsyncNimble` for async
- Implement both `_get_relevant_documents()` and `_aget_relevant_documents()`
- For tools: implement both `_run()` and `_arun()`

### Client Initialization
- Clients initialized once in `@model_validator(mode="after")`
- Reuse clients across requests (connection pooling)
- All tools/retrievers extend `_NimbleClientMixin` from `_utilities.py`

### Error Handling
- SDK handles retries on 5xx (configured via `max_retries` on client)
- `handle_api_errors()` context manager converts SDK exceptions to `ToolException`
- Raise `ToolException` (not `ValueError`) for graceful agent error handling

### Tool Patterns
- All tools extend `_NimbleClientMixin, BaseTool` and define `args_schema`, `handle_tool_error = True`
- Each tool has `_build_*_kwargs()` to construct SDK call params, `_run()` sync, `_arun()` async
- For async SDK operations (crawl): use poll-inside-the-tool with `time.monotonic()` deadline
- NimbleToolkit groups tools with `include_*` flags; `get_tools()` returns `list[BaseTool]`
- Extract Templates: 3 tools (list→get→run) via `include_extract_templates`
- Agent API V2: resumable start/status/result (no poll-inside-tool) via `include_web_search_agents`
- Deprecated `NimbleAgent*` / `include_agent` wrap Extract Templates only — never Agent API V2
- Attribution: `client_source="langchain-nimble"` → `X-Client-Source: langchain-nimble`

### Nimble SDK Introspection
- Inspect SDK methods: `uv run python -c "from nimble_python import Nimble; import inspect; print(inspect.signature(Nimble(api_key='x').search))"`
- Inspect response types: `uv run python -c "from nimble_python.types import SearchResponse; print(SearchResponse.model_fields)"`
- Prefer `nimble_python>=1.2.0`: `search()`, `extract.run()`, `extract.templates.*`, `agents.*` (typed `agent_name`/`use_case`/`skill` on runs), `map()`; `crawl.run()` needs polling via `crawl.status()`
- Agent API V2: `agents.runs.create` / `get` / `result` (resumable across turns)

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

# Local (use absolute imports — ruff TID252 forbids relative parent imports)
from langchain_nimble._utilities import _NimbleClientMixin, handle_api_errors
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

---

## Releasing

Releases to PyPI are automated. **Never build and upload by hand** - publishing a
GitHub Release is the only supported path, and it is what produces the artifacts
users install.

### Steps

1. Bump `version` in `pyproject.toml`, following semver.
2. Open a PR with that bump and merge it to `main`.
3. Tag the merge commit `vX.Y.Z` and publish a GitHub Release from that tag.

The `Release` workflow (`.github/workflows/release.yml`) then verifies the tag
against `pyproject.toml`, builds the sdist and wheel, and uploads to PyPI.

### Rules

- **The tag must match the version exactly**, with a `v` prefix: `pyproject.toml`
  `version = "3.1.0"` requires tag `v3.1.0`. A mismatch fails the run before the
  build step, so it never reaches PyPI.
- **No PyPI credentials are involved.** The workflow authenticates via OIDC
  Trusted Publishing, which mints a short-lived, project-scoped token at publish
  time. There is no token to configure, rotate, or leak. Do not add one.
- **Tags matching `v*` are protected** and require elevated repository
  permissions. A rejected tag push means you need a maintainer to cut the
  release, not a workaround.
- **The workflow body executes from the tag's tree**, not from `main`. A release
  tag therefore pins both the code and the release process itself, which is why
  the tag must point at a reviewed, merged commit.
- **Actions in the publish job are pinned to full commit SHAs.** When bumping
  one, update the trailing version comment in the same edit.

### Before tagging

```bash
make lint          # ruff + mypy
make test          # unit tests, sockets disabled
make check_imports # verify all public imports resolve
```

Integration tests need `NIMBLE_API_KEY` and are worth running when the release
touches request construction or response parsing:

```bash
make integration_tests
```

### After publishing

Confirm the new version appears on
[PyPI](https://pypi.org/project/langchain-nimble/) and that the release carries
provenance attestations. If the run failed, fix forward with a new patch version
rather than deleting or moving the tag - published versions cannot be replaced on
PyPI, and release tags are immutable by policy.

