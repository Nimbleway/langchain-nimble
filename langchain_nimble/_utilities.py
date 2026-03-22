"""Nimble SDK client utilities."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from langchain_core.utils import secret_from_env
from nimble_python import APIConnectionError as NimbleConnectionError
from nimble_python import APIStatusError as NimbleStatusError
from nimble_python import APITimeoutError as NimbleTimeoutError
from nimble_python import AsyncNimble, Nimble
from pydantic import BaseModel, Field, SecretStr, model_validator


class _NimbleClientMixin(BaseModel):
    """Mixin providing Nimble API client configuration and initialization.

    This mixin is shared by both retrievers and tools to avoid code duplication
    for client configuration and initialization logic.
    """

    nimble_api_key: SecretStr = Field(
        alias="api_key",
        default_factory=secret_from_env("NIMBLE_API_KEY", default=""),
    )
    nimble_api_url: str | None = Field(
        alias="base_url",
        default=None,
        description="Override base URL for the Nimble API (default: SDK default).",
    )
    max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
        description="Maximum retry attempts for 5xx errors (0 disables retries)",
    )

    locale: str = "en"
    country: str = "US"
    output_format: str = "markdown"

    _sync_client: Nimble | None = None
    _async_client: AsyncNimble | None = None

    @model_validator(mode="after")
    def initialize_clients(self) -> _NimbleClientMixin:
        """Initialize Nimble SDK clients."""
        api_key = self.nimble_api_key.get_secret_value()
        if not api_key:
            msg = "API key required. Set NIMBLE_API_KEY or pass api_key parameter."
            raise ValueError(msg)

        client_kwargs: dict[str, object] = {
            "api_key": api_key,
            "max_retries": self.max_retries,
            "default_headers": {"X-Client-Source": "langchain-nimble"},
        }
        if self.nimble_api_url is not None:
            client_kwargs["base_url"] = self.nimble_api_url

        self._sync_client = Nimble(**client_kwargs)  # type: ignore[arg-type]
        self._async_client = AsyncNimble(**client_kwargs)  # type: ignore[arg-type]
        return self


@contextmanager
def handle_api_errors(operation: str = "API request") -> Iterator[None]:
    """Convert Nimble SDK exceptions to user-friendly error messages."""
    try:
        yield
    except NimbleStatusError as e:
        status = e.status_code
        if 400 <= status < 500:
            msg = (
                f"Nimble API {operation} failed with client error ({status}): "
                f"{e.message}"
            )
        else:
            msg = (
                f"Nimble API {operation} failed with server error ({status}): "
                f"{e.message}"
            )
        raise ValueError(msg) from e
    except NimbleTimeoutError as e:
        msg = f"Nimble API {operation} timed out: {e.message}"
        raise ValueError(msg) from e
    except NimbleConnectionError as e:
        msg = f"Nimble API {operation} failed with network error: {e.message}"
        raise ValueError(msg) from e
