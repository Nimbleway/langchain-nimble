"""Unit tests for Nimble Extract Template tools."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.tools import ToolException

from langchain_nimble import (
    NimbleExtractTemplateGetTool,
    NimbleExtractTemplateListTool,
    NimbleExtractTemplateRunTool,
)


def _mock_template_list_response() -> MagicMock:
    """Create a mock TemplateListResponse with .items."""
    item1 = MagicMock()
    item1.model_dump.return_value = {"name": "amazon_pdp", "id": "tmpl_1"}
    item2 = MagicMock()
    item2.model_dump.return_value = {"name": "google_search", "id": "tmpl_2"}
    response = MagicMock()
    response.items = [item1, item2]
    return response


def test_extract_template_list_init() -> None:
    """Test NimbleExtractTemplateListTool initialization."""
    tool = NimbleExtractTemplateListTool(api_key="test_key")
    assert tool.name == "nimble_extract_template_list"


def test_extract_template_list_run() -> None:
    """Test synchronous extract template list."""
    tool = NimbleExtractTemplateListTool(api_key="test_key")
    mock_response = _mock_template_list_response()

    with patch.object(
        tool._sync_client.extract.templates,
        "list",
        return_value=mock_response,
    ) as mock_list:
        result = tool._run(limit=10, offset=0)

    assert len(result) == 2
    assert result[0]["name"] == "amazon_pdp"
    mock_list.assert_called_once_with(limit=10, offset=0)


async def test_extract_template_list_arun() -> None:
    """Test asynchronous extract template list."""
    tool = NimbleExtractTemplateListTool(api_key="test_key")
    mock_response = _mock_template_list_response()

    with patch.object(
        tool._async_client.extract.templates,
        "list",
        return_value=mock_response,
    ) as mock_list:
        result = await tool._arun()

    assert len(result) == 2
    mock_list.assert_awaited_once()


def test_extract_template_get_run() -> None:
    """Test synchronous extract template get."""
    tool = NimbleExtractTemplateGetTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"name": "amazon_pdp", "id": "tmpl_1"}

    with patch.object(
        tool._sync_client.extract.templates,
        "get",
        return_value=mock_response,
    ) as mock_get:
        result = tool._run(template_name="amazon_pdp")

    assert result["name"] == "amazon_pdp"
    mock_get.assert_called_once_with("amazon_pdp")


async def test_extract_template_get_arun() -> None:
    """Test asynchronous extract template get."""
    tool = NimbleExtractTemplateGetTool(api_key="test_key")
    mock_response = MagicMock()
    mock_response.model_dump.return_value = {"name": "amazon_pdp"}

    with patch.object(
        tool._async_client.extract.templates,
        "get",
        return_value=mock_response,
    ) as mock_get:
        result = await tool._arun(template_name="amazon_pdp")

    assert result["name"] == "amazon_pdp"
    mock_get.assert_awaited_once()


def _mock_template_run_response(*, status: str = "success") -> MagicMock:
    """Create a mock TemplateRunResponse."""
    mock = MagicMock()
    mock.status = status
    mock.task_id = "task_123"
    mock.warnings = None
    mock.model_dump.return_value = {
        "status": status,
        "task_id": "task_123",
        "data": {"title": "Product"},
    }
    return mock


def test_extract_template_run_success() -> None:
    """Test successful extract template run."""
    tool = NimbleExtractTemplateRunTool(api_key="test_key")
    mock_response = _mock_template_run_response()

    with patch.object(
        tool._sync_client.extract.templates,
        "run",
        return_value=mock_response,
    ) as mock_run:
        result = tool._run(template="amazon_pdp", params={"asin": "B0TEST"})

    assert result["status"] == "success"
    mock_run.assert_called_once()
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["template"] == "amazon_pdp"
    assert call_kwargs["params"] == {"asin": "B0TEST"}


async def test_extract_template_run_arun() -> None:
    """Test asynchronous extract template run."""
    tool = NimbleExtractTemplateRunTool(api_key="test_key")
    mock_response = _mock_template_run_response()

    with patch.object(
        tool._async_client.extract.templates,
        "run",
        return_value=mock_response,
    ) as mock_run:
        result = await tool._arun(
            template="amazon_pdp",
            params={"asin": "B0TEST"},
            localization=True,
        )

    assert result["status"] == "success"
    call_kwargs = mock_run.call_args.kwargs
    assert call_kwargs["localization"] is True


def test_extract_template_run_failure_status() -> None:
    """Test extract template run raises on non-success status."""
    tool = NimbleExtractTemplateRunTool(api_key="test_key")
    mock_response = _mock_template_run_response(status="fatal")
    mock_response.warnings = ["bad input"]

    with (
        patch.object(
            tool._sync_client.extract.templates,
            "run",
            return_value=mock_response,
        ),
        pytest.raises(ToolException, match="fatal"),
    ):
        tool._run(template="amazon_pdp", params={"asin": "B0TEST"})
