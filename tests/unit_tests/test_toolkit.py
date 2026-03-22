"""Unit tests for NimbleToolkit."""

from langchain_nimble import NimbleToolkit
from langchain_nimble.tools.agent_tool import (
    NimbleAgentGetTool,
    NimbleAgentListTool,
    NimbleAgentRunTool,
)
from langchain_nimble.tools.crawl_tool import NimbleCrawlTool
from langchain_nimble.tools.extract_tool import NimbleExtractTool
from langchain_nimble.tools.map_tool import NimbleMapTool
from langchain_nimble.tools.search_tool import NimbleSearchTool


def test_toolkit_default_tools() -> None:
    """Test toolkit returns Search + Extract by default."""
    toolkit = NimbleToolkit(api_key="test_key")
    tools = toolkit.get_tools()

    assert len(tools) == 2
    tool_types = {type(t) for t in tools}
    assert tool_types == {NimbleSearchTool, NimbleExtractTool}


def test_toolkit_all_tools() -> None:
    """Test toolkit returns all 7 tools when all flags enabled."""
    toolkit = NimbleToolkit(
        api_key="test_key",
        include_crawl=True,
        include_map=True,
        include_agent=True,
    )
    tools = toolkit.get_tools()

    assert len(tools) == 7
    tool_types = {type(t) for t in tools}
    assert tool_types == {
        NimbleSearchTool,
        NimbleExtractTool,
        NimbleCrawlTool,
        NimbleMapTool,
        NimbleAgentListTool,
        NimbleAgentGetTool,
        NimbleAgentRunTool,
    }


def test_toolkit_selective_inclusion() -> None:
    """Test toolkit with selective flags."""
    toolkit = NimbleToolkit(
        api_key="test_key",
        include_search=False,
        include_extract=False,
        include_map=True,
    )
    tools = toolkit.get_tools()

    assert len(tools) == 1
    assert isinstance(tools[0], NimbleMapTool)


def test_toolkit_no_tools() -> None:
    """Test toolkit with all flags disabled."""
    toolkit = NimbleToolkit(
        api_key="test_key",
        include_search=False,
        include_extract=False,
    )
    tools = toolkit.get_tools()

    assert len(tools) == 0


def test_toolkit_passes_api_key() -> None:
    """Test toolkit passes API key to all tools."""
    toolkit = NimbleToolkit(
        api_key="shared_key",
        include_crawl=True,
        include_map=True,
        include_agent=True,
    )
    tools = toolkit.get_tools()

    for tool in tools:
        assert tool.nimble_api_key.get_secret_value() == "shared_key"  # type: ignore[union-attr]


def test_toolkit_passes_base_url() -> None:
    """Test toolkit passes base_url to all tools."""
    toolkit = NimbleToolkit(
        api_key="test_key",
        base_url="https://custom.api.com",
        include_map=True,
    )
    tools = toolkit.get_tools()

    for tool in tools:
        assert tool.nimble_api_url == "https://custom.api.com"  # type: ignore[union-attr]


def test_toolkit_passes_crawl_config() -> None:
    """Test toolkit passes crawl-specific config to crawl tool."""
    toolkit = NimbleToolkit(
        api_key="test_key",
        include_crawl=True,
        crawl_polling_interval=2.0,
        crawl_timeout=120.0,
    )
    tools = toolkit.get_tools()

    crawl_tools = [t for t in tools if isinstance(t, NimbleCrawlTool)]
    assert len(crawl_tools) == 1
    assert crawl_tools[0].polling_interval == 2.0
    assert crawl_tools[0].timeout == 120.0


def test_toolkit_agent_includes_three_tools() -> None:
    """Test include_agent adds list, get, and run tools."""
    toolkit = NimbleToolkit(
        api_key="test_key",
        include_search=False,
        include_extract=False,
        include_agent=True,
    )
    tools = toolkit.get_tools()

    assert len(tools) == 3
    tool_types = {type(t) for t in tools}
    assert tool_types == {
        NimbleAgentListTool,
        NimbleAgentGetTool,
        NimbleAgentRunTool,
    }


def test_toolkit_tool_names() -> None:
    """Test all tools have expected names."""
    toolkit = NimbleToolkit(
        api_key="test_key",
        include_crawl=True,
        include_map=True,
        include_agent=True,
    )
    tools = toolkit.get_tools()
    tool_names = {t.name for t in tools}

    assert tool_names == {
        "nimble_search",
        "nimble_extract",
        "nimble_crawl",
        "nimble_map",
        "nimble_agent_list",
        "nimble_agent_get",
        "nimble_agent_run",
    }
