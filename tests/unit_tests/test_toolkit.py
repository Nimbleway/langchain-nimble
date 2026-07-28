"""Unit tests for NimbleToolkit."""

import pytest

from langchain_nimble import NimbleToolkit
from langchain_nimble.tools.agent_tool import (
    NimbleAgentGetTool,
    NimbleAgentListTool,
    NimbleAgentRunTool,
)
from langchain_nimble.tools.agents_v2_tool import (
    NimbleAgentCreateTool,
    NimbleAgentRunResultTool,
    NimbleAgentRunStartTool,
    NimbleAgentRunStatusTool,
    NimbleAgentsListTool,
    NimbleAgentTemplatesListTool,
)
from langchain_nimble.tools.crawl_tool import NimbleCrawlTool
from langchain_nimble.tools.extract_template_tool import (
    NimbleExtractTemplateGetTool,
    NimbleExtractTemplateListTool,
    NimbleExtractTemplateRunTool,
)
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
    """Test toolkit returns all tools when all modern flags enabled."""
    toolkit = NimbleToolkit(
        api_key="test_key",
        include_crawl=True,
        include_map=True,
        include_extract_templates=True,
        include_web_search_agents=True,
    )
    tools = toolkit.get_tools()

    assert len(tools) == 13
    tool_types = {type(t) for t in tools}
    assert tool_types == {
        NimbleSearchTool,
        NimbleExtractTool,
        NimbleCrawlTool,
        NimbleMapTool,
        NimbleExtractTemplateListTool,
        NimbleExtractTemplateGetTool,
        NimbleExtractTemplateRunTool,
        NimbleAgentsListTool,
        NimbleAgentTemplatesListTool,
        NimbleAgentCreateTool,
        NimbleAgentRunStartTool,
        NimbleAgentRunStatusTool,
        NimbleAgentRunResultTool,
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
        include_extract_templates=True,
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


def test_toolkit_include_extract_templates() -> None:
    """Test include_extract_templates adds three template tools."""
    toolkit = NimbleToolkit(
        api_key="test_key",
        include_search=False,
        include_extract=False,
        include_extract_templates=True,
    )
    tools = toolkit.get_tools()

    assert len(tools) == 3
    tool_types = {type(t) for t in tools}
    assert tool_types == {
        NimbleExtractTemplateListTool,
        NimbleExtractTemplateGetTool,
        NimbleExtractTemplateRunTool,
    }


def test_toolkit_include_web_search_agents() -> None:
    """Test include_web_search_agents adds six Agent API V2 tools."""
    toolkit = NimbleToolkit(
        api_key="test_key",
        include_search=False,
        include_extract=False,
        include_web_search_agents=True,
    )
    tools = toolkit.get_tools()

    assert len(tools) == 6
    tool_names = {t.name for t in tools}
    assert tool_names == {
        "nimble_web_search_agents_list",
        "nimble_web_search_agent_templates_list",
        "nimble_web_search_agent_create",
        "nimble_web_search_agent_run_start",
        "nimble_web_search_agent_run_status",
        "nimble_web_search_agent_run_result",
    }


def test_toolkit_deprecated_include_agent() -> None:
    """Test include_agent warns and returns deprecated aliases."""
    toolkit = NimbleToolkit(
        api_key="test_key",
        include_search=False,
        include_extract=False,
        include_agent=True,
    )

    with pytest.warns(DeprecationWarning, match="include_agent"):
        tools = toolkit.get_tools()

    assert len(tools) == 3
    tool_types = {type(t) for t in tools}
    assert tool_types == {
        NimbleAgentListTool,
        NimbleAgentGetTool,
        NimbleAgentRunTool,
    }


def test_toolkit_extract_templates_preferred_over_include_agent() -> None:
    """Test include_extract_templates wins over deprecated include_agent."""
    toolkit = NimbleToolkit(
        api_key="test_key",
        include_search=False,
        include_extract=False,
        include_extract_templates=True,
        include_agent=True,
    )
    with pytest.warns(DeprecationWarning, match="include_agent"):
        tools = toolkit.get_tools()

    assert len(tools) == 3
    tool_names = {t.name for t in tools}
    assert tool_names == {
        "nimble_extract_template_list",
        "nimble_extract_template_get",
        "nimble_extract_template_run",
    }


def test_toolkit_include_agent_and_web_search_agents_no_aliases() -> None:
    """Test dual flags warn and only V2 tools appear (no deprecated aliases)."""
    toolkit = NimbleToolkit(
        api_key="test_key",
        include_search=False,
        include_extract=False,
        include_agent=True,
        include_web_search_agents=True,
    )

    with pytest.warns(DeprecationWarning) as recorded:
        tools = toolkit.get_tools()

    warning_text = " ".join(str(w.message) for w in recorded)
    assert "include_agent" in warning_text
    assert "nimble_agent_*" in warning_text or "ignored" in warning_text

    tool_names = {t.name for t in tools}
    assert tool_names == {
        "nimble_web_search_agents_list",
        "nimble_web_search_agent_templates_list",
        "nimble_web_search_agent_create",
        "nimble_web_search_agent_run_start",
        "nimble_web_search_agent_run_status",
        "nimble_web_search_agent_run_result",
    }
    assert "nimble_agent_list" not in tool_names
    assert "nimble_agent_run" not in tool_names


def test_toolkit_tool_names() -> None:
    """Test tools have expected names with modern flags."""
    toolkit = NimbleToolkit(
        api_key="test_key",
        include_crawl=True,
        include_map=True,
        include_extract_templates=True,
        include_web_search_agents=True,
    )
    tools = toolkit.get_tools()
    tool_names = {t.name for t in tools}

    assert tool_names == {
        "nimble_search",
        "nimble_extract",
        "nimble_crawl",
        "nimble_map",
        "nimble_extract_template_list",
        "nimble_extract_template_get",
        "nimble_extract_template_run",
        "nimble_web_search_agents_list",
        "nimble_web_search_agent_templates_list",
        "nimble_web_search_agent_create",
        "nimble_web_search_agent_run_start",
        "nimble_web_search_agent_run_status",
        "nimble_web_search_agent_run_result",
    }
