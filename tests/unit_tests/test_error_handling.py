"""Unit tests for error handling."""

from unittest.mock import Mock

import httpx
import pytest

from langchain_nimble._utilities import handle_api_errors


class TestErrorHandling:
    """Test error handling produces helpful messages."""

    def test_client_error_4xx(self) -> None:
        """Test 4xx errors produce client error messages."""
        with (
            pytest.raises(ValueError, match=r"client error.*401"),
            handle_api_errors("test operation"),
        ):
            response = Mock()
            response.status_code = 401
            response.text = "Unauthorized"
            msg = "401 Unauthorized"
            raise httpx.HTTPStatusError(
                msg,
                request=Mock(),
                response=response,
            )

    def test_server_error_5xx(self) -> None:
        """Test 5xx errors produce server error messages."""
        with (
            pytest.raises(ValueError, match=r"server error.*503"),
            handle_api_errors("test operation"),
        ):
            response = Mock()
            response.status_code = 503
            response.text = "Service Unavailable"
            msg = "503 Service Unavailable"
            raise httpx.HTTPStatusError(
                msg,
                request=Mock(),
                response=response,
            )

    def test_timeout_error(self) -> None:
        """Test timeout errors produce helpful messages."""
        with (
            pytest.raises(ValueError, match="timed out"),
            handle_api_errors("test operation"),
        ):
            msg = "Request timeout"
            raise httpx.TimeoutException(msg)

    def test_network_error(self) -> None:
        """Test network errors produce helpful messages."""
        with (
            pytest.raises(ValueError, match="network error"),
            handle_api_errors("test operation"),
        ):
            msg = "Connection failed"
            raise httpx.RequestError(msg)
