"""Unit tests for error handling."""

from unittest.mock import Mock

import pytest
from langchain_core.tools import ToolException
from nimble_python import APIConnectionError, APIStatusError, APITimeoutError

from langchain_nimble._utilities import handle_api_errors


class TestErrorHandling:
    """Test error handling produces helpful messages."""

    def test_client_error_4xx(self) -> None:
        """Test 4xx errors produce client error messages."""
        with (
            pytest.raises(ToolException, match=r"client error.*401"),
            handle_api_errors("test operation"),
        ):
            response = Mock()
            response.status_code = 401
            response.headers = {}
            msg = "Unauthorized"
            raise APIStatusError(
                msg,
                response=response,
                body=None,
            )

    def test_server_error_5xx(self) -> None:
        """Test 5xx errors produce server error messages."""
        with (
            pytest.raises(ToolException, match=r"server error.*503"),
            handle_api_errors("test operation"),
        ):
            response = Mock()
            response.status_code = 503
            response.headers = {}
            msg = "Service Unavailable"
            raise APIStatusError(
                msg,
                response=response,
                body=None,
            )

    def test_timeout_error(self) -> None:
        """Test timeout errors produce helpful messages."""
        with (
            pytest.raises(ToolException, match="timed out"),
            handle_api_errors("test operation"),
        ):
            request = Mock()
            raise APITimeoutError(request=request)

    def test_network_error(self) -> None:
        """Test network errors produce helpful messages."""
        with (
            pytest.raises(ToolException, match="network error"),
            handle_api_errors("test operation"),
        ):
            request = Mock()
            raise APIConnectionError(request=request)
