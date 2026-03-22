"""Shared type definitions for Nimble Search API."""

from enum import Enum


class SearchFocus(str, Enum):
    """Search focus mode/specialization."""

    GENERAL = "general"
    NEWS = "news"
    LOCATION = "location"
    SHOPPING = "shopping"
    GEO = "geo"
    SOCIAL = "social"


class SearchDepth(str, Enum):
    """Search depth level controlling content retrieval."""

    LITE = "lite"
    FAST = "fast"
    DEEP = "deep"


class OutputFormat(str, Enum):
    """Content output format."""

    PLAIN_TEXT = "plain_text"
    MARKDOWN = "markdown"
    SIMPLIFIED_HTML = "simplified_html"


OUTPUT_FORMAT_TO_SDK_FORMATS: dict[str, list[str]] = {
    "markdown": ["markdown"],
    "plain_text": ["html"],
    "simplified_html": ["html"],
}


class BrowserlessDriver(str, Enum):
    """Browserless drivers available for web extraction."""

    VX6 = "vx6"
    VX8 = "vx8"
    VX8_PRO = "vx8-pro"
    VX10 = "vx10"
    VX10_PRO = "vx10-pro"
    VX12 = "vx12"
    VX12_PRO = "vx12-pro"
